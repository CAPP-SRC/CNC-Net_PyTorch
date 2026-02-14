import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import utils
import utils.workspace as ws
from utils.eval_metric import IOU
from utils.cad_meshing	import create_mesh_mc
from networks.model import Model
import utils.dataloader as dataloader
from utils.Logger import Logger
import utils.utils as utils



class LearningRateSchedule:
	def get_learning_rate(self, epoch):
		pass


class ConstantLearningRateSchedule(LearningRateSchedule):
	def __init__(self, value):
		self.value = value

	def get_learning_rate(self, epoch):
		return self.value


class StepLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, interval, factor):
		self.initial = initial
		self.interval = interval
		self.factor = factor

	def get_learning_rate(self, epoch):

		return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, warmed_up, length):
		self.initial = initial
		self.warmed_up = warmed_up
		self.length = length

	def get_learning_rate(self, epoch):
		if epoch > self.length:
			return self.warmed_up
		return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

	schedule_specs = specs["LearningRateSchedule"]
	print(schedule_specs)

	schedules = []

	for schedule_specs in schedule_specs:
		if schedule_specs["Type"] == "Step":

			schedules.append(
				StepLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Interval"],
					schedule_specs["Factor"],
				)
			)
		elif schedule_specs["Type"] == "Warmup":
			schedules.append(
				WarmupLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Final"],
					schedule_specs["Length"],
				)
			)
		elif schedule_specs["Type"] == "Constant":
			schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

		else:
			raise Exception(
				'no known learning rate schedule of type "{}"'.format(
					schedule_specs["Type"]
				)
			)

	return schedules

def get_spec_with_default(specs, key, default):
	try:
		return specs[key]
	except KeyError:
		return default

def init_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def _train_fn(experiment_directory, continue_from, input_object):
	"""Per-device training function launched by torch_xla.launch.

	Each process owns exactly one XLA device (one TPU chip).
	On a v4-32 pod there will be 16 processes across 4 hosts.
	"""
	device = xm.xla_device()
	ordinal = xm.get_ordinal()
	is_master = xm.is_master_ordinal()

	init_seeds(seed=ordinal)

	out_dir = 'checkpoints/' + experiment_directory + '/'

	if is_master:
		logger = logging.getLogger()
		handler = logging.FileHandler(out_dir + 'logfile.log')
		logger.addHandler(handler)
		logger.debug("running " + out_dir)

	specs = ws.load_experiment_specifications('configs')

	if is_master:
		logging.info("Experiment description: \n" + specs["Description"])

	arch = __import__("networks." + specs["NetworkArch"], fromlist=["PolyNet", "Decoder"])

	checkpoints = list(
		range(
			specs["SnapshotFrequency"],
			specs["NumEpochs"] + 1,
			specs["SnapshotFrequency"],
		)
	)

	for checkpoint in specs["AdditionalSnapshots"]:
		checkpoints.append(checkpoint)
	checkpoints.sort()
	if is_master:
		print(checkpoints)
	lr_schedules = get_learning_rate_schedules(specs)


	def save_checkpoints(epoch):
		xm.save(
			{
				'epoch': epoch,
				'operation_state_dict': operation.state_dict(),
				'optimizer_state_dict': optimizer_operation.state_dict(),
			},
			os.path.join(out_dir, str(epoch) + ".pth"),
		)


	def signal_handler(sig, frame):
		logging.info("Stopping early...")
		sys.exit(0)

	def adjust_learning_rate(lr_schedules, optimizer, epoch):

		for i, param_group in enumerate(optimizer.param_groups):
			param_group["lr"] = lr_schedules[0].get_learning_rate(epoch)
			if is_master:
				print(param_group["lr"])

	start_time = time.time()
	signal.signal(signal.SIGINT, signal_handler)


	operation = Model(ef_dim=256)
	operation = operation.to(device)



	num_epochs = specs["NumEpochs"]
	mse = nn.MSELoss(reduction='mean')



	if input_object is not None:
		occ_dataset_train = dataloader.DataLoader(
			test_flag=True, inpt=input_object
		)
		occ_dataset_test = dataloader.DataLoader(
			test_flag=True, inpt=input_object
		)



	train_loader = pl.MpDeviceLoader(
		data_utils.DataLoader(
			occ_dataset_train,
			batch_size=1,
			shuffle=False,
			num_workers=4,
		),
		device,
	)

	test_loader = pl.MpDeviceLoader(
		data_utils.DataLoader(
			occ_dataset_test,
			batch_size=1,
			shuffle=False,
			num_workers=4,
		),
		device,
	)


	num_scenes = len(occ_dataset_train)
	if is_master:
		logging.info("There are {} shapes".format(num_scenes))

	logging.debug(operation)
	optimizer_operation = torch.optim.Adam(
		[
			{
				"params": (operation.parameters()),
				"lr": lr_schedules[0].get_learning_rate(0),
				"betas": (0.5, 0.999),
			},
		]
	)



	start_epoch = 0
	if continue_from is not None:

		if is_master:
			logging.info('continuing from "{}"'.format(continue_from))
		load = torch.load(
			out_dir + 'operation_checkpoint_' + str(continue_from) + '.pth',
			map_location='cpu',
		)
		operation.load_state_dict(load["operation_state_dict"])
		optimizer_operation.load_state_dict(load["optimizer_state_dict"])
		model_epoch = load["epoch"]
		start_epoch = model_epoch + 1
		# Re-move model to device after loading CPU state dict
		operation = operation.to(device)
		logging.debug("loaded")
		if is_master:
			logging.info("starting from epoch {}".format(start_epoch))

	operation.train()




	last_epoch_time = 0
	best_iou = torch.zeros(len(train_loader))



	# Training
	BEST_IOU = 0
	for epoch in range(start_epoch, start_epoch + num_epochs):


		adjust_learning_rate(lr_schedules, optimizer_operation, epoch - start_epoch)

		TOTAL_LOSS = 0
		for inds_inout, all_points, all_points_high, dimension, shape_names in tqdm(train_loader, disable=not is_master):
			# Data is already on the XLA device (MpDeviceLoader handles transfer).
			# MpDeviceLoader calls torch_xla.sync() when yielding the next batch.
			with torch_xla.step():
				current = -torch.ones_like(inds_inout)
				current_high = -torch.ones(1, 256*256*256, device=device).float()

				total_loss, outputs = operation(current, current_high, all_points, all_points_high, inds_inout, 100, out_dir, torch.mean(best_iou.detach()), dimension, epoch, False)


				if not math.isnan(total_loss):
					TOTAL_LOSS += total_loss.detach() / len(train_loader)


				optimizer_operation.zero_grad()
				total_loss.backward()
				optimizer_operation.step()

			del total_loss
			del outputs




		if (epoch-start_epoch+1) in checkpoints:
			save_checkpoints(epoch)

		# Testing
		if (epoch+1) % 1 == 0:
			#operation.eval()
			IOU_total = []
			with torch.no_grad():
				# Data is already on the XLA device (MpDeviceLoader handles transfer).
				inds_inout, all_points, all_points_high, dimension, shape_names = next(iter(test_loader))

				current = -torch.ones_like(inds_inout)
				current_high = -torch.ones(1, 256*256*256, device=device).float()

				_, outputs, outputs_high = operation(current, current_high, all_points, all_points_high, inds_inout, 100, out_dir, best_iou[shape_names], dimension, epoch, True)
				iou = IOU(outputs, inds_inout)
				IOU_total.append(iou)
				if best_iou[shape_names]<iou:
					best_iou[shape_names]=iou

				outputs_high = 0.5*(-torch.sign(outputs_high)+1)
				samples = all_points_high

				if is_master:
					create_mesh_mc(samples, outputs_high, dimension, os.path.join("output", experiment_directory, input_object[:-4]))

				average_best_iou = sum(IOU_total)/len(IOU_total)
			if is_master:
				logging.debug('Average IOU:\t{:.6f}'.format(average_best_iou))


		if BEST_IOU < average_best_iou:

			BEST_IOU = average_best_iou
			model_path = out_dir + "operation_checkpoint_" + str(epoch) + ".pth"
			xm.save({
				'epoch': epoch,
				'operation_state_dict': operation.state_dict(),
				'optimizer_state_dict': optimizer_operation.state_dict(),
			}, model_path)
			model_path2 = out_dir + "best_checkpoint" + ".pth"
			xm.save({
				'epoch': epoch,
				'operation_state_dict': operation.state_dict(),
				'optimizer_state_dict': optimizer_operation.state_dict(),
			}, model_path2)
			if is_master:
				logging.debug('BEST IOU:\t{:.6f}'.format(BEST_IOU))

			seconds_elapsed = time.time() - start_time
			ava_epoch_time = (seconds_elapsed - last_epoch_time)/10
			last_epoch_time = seconds_elapsed


		if is_master:
			logging.debug("epoch = {}/{} , \
				total_loss={:.6f}".format(epoch, num_epochs+start_epoch, TOTAL_LOSS))



if __name__ == "__main__":

	import argparse

	arg_parser = argparse.ArgumentParser(description="Train a Network on TPU")
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True,
		help="The experiment directory. This directory should include "
		+ "experiment specifications in 'specs.json', and logging will be "
		+ "done in this directory as well.",
	)
	arg_parser.add_argument(
		"--continue",
		"-c",
		dest="continue_from",
		help="A snapshot to continue from. This can be an integer corresponding to an epochal snapshot.",
	)

	arg_parser.add_argument(
		"--input_object",
		"-i",
		dest = "input_object",
		required = True,
		help = "The object name"

	)


	utils.add_common_args(arg_parser)

	args = arg_parser.parse_args()

	utils.configure_logging(args)

	if args.input_object == 'all':
		args.input_object = None

	directory_path = "checkpoints/" + str(args.experiment_directory)

	# Check if the directory already exists
	if not os.path.exists(directory_path):
		# If it doesn't exist, create it
		os.makedirs(directory_path)
		print(f"Directory '{directory_path}' created.")
	else:
		print(f"Directory '{directory_path}' already exists.")


	directory_path = "output/" + str(args.experiment_directory)
	# Check if the directory already exists
	if not os.path.exists(directory_path):
		# If it doesn't exist, create it
		os.makedirs(directory_path)
		print(f"Directory '{directory_path}' created.")
	else:
		print(f"Directory '{directory_path}' already exists.")

	# Launch training across all available TPU devices.
	# On a v4-32 pod (16 chips / 4 hosts) PJRT spawns one process per chip.
	torch_xla.launch(
		_train_fn,
		args=(args.experiment_directory, args.continue_from, args.input_object),
	)
