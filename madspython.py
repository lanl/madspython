#! /usr/bin/env python

"""
Copyright (2016).  Los Alamos National Security, LLC. This material was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
Additionally, this program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. Accordingly, this program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
"""

import yaml
import multiprocessing
import subprocess
import copy
import sys
import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
#import optinoise

from collections import OrderedDict

class UnsortableList(list):
    def sort(self, *args, **kwargs):
        pass

class UnsortableOrderedDict(OrderedDict):
	def items(self, *args, **kwargs):
		return UnsortableList(OrderedDict.items(self, *args, **kwargs))

yaml.add_representer(UnsortableOrderedDict, yaml.representer.SafeRepresenter.represent_dict)

mads_bin = 'mads' #TODO I will suggest to use the global search path; "mads" may be good enough

class MadsOrderedParams:
	paramorderdict = {"x": 1, "y": 2, "z": 3, "dx": 4, "dy": 5, "dz": 6, "f": 7, "t0": 8, "t1": 9, "n": 10, "rf": 11, "lambda": 12, "theta": 13, "vx": 14, "vy": 15, "vz": 16, "ax": 17, "ay": 18, "az": 19, "ts_dsp": 20, "ts_adv": 21, "ts_rct": 22, "alpha": 23, "beta": 24, "nlc0": 25, "nlc1": 26}
	def __init__(self, paramnames, paramvals):
		od = dict(zip(map(lambda x: MadsOrderedParams.paramorderdict[x], paramnames), paramvals))
		skeys = sorted(od)
		self.vals = map(lambda i: od[i], skeys)
	def __str__(self):
		return " ".join(map(str, self.vals))

def betmads_sampling(paramnames, paramvalsarray, inmadsfilename, outmadsfilename, save=False):
	outfilenameroot = ".".join(outmadsfilename.split(".")[:-1])
	my = MadsYaml(inmadsfilename)
	for i in range(len(paramnames)):
		my.setParamType(paramnames[i], 2)
	my.dump(outmadsfilename)
	with open(outfilenameroot + ".igrnd.results", "w") as f:
		for i in range(len(paramvalsarray)):
			mop = MadsOrderedParams(paramnames, paramvalsarray[i])
			f.write(str(i + 1) + " : init var " + str(mop) + " : OF 0.0 success 0 : final var " + str(mop) + "\n")
		f.close()
	if save:
		mads_exec(outmadsfilename + " resultsfile=" + outfilenameroot + ".igrnd.results resultscase=-" + str(len(paramvalsarray)) + " save force")
	else:
		mads_exec(outmadsfilename + " resultsfile=" + outfilenameroot + ".igrnd.results resultscase=-" + str(len(paramvalsarray)) + " force")
	with open(outfilenameroot + ".phi") as f:
		oflines = f.readlines()
		f.close()
	if save:
		numobs = my.getNumObs()
		data = np.zeros((len(paramvalsarray), numobs + 1))
		for i in range(len(paramvalsarray)):
			data[i][0] = float(oflines[i])#get the objective function
			with open(outfilenameroot + "." + str(i + 1) + ".results") as f:
				resultslines = f.readlines()
				startline = 0
				while not ("Model predictions" in resultslines[startline]):
					startline += 1
				startline += 1
				for j in range(numobs):
					data[i][j + 1] = float(resultslines[startline + j].split(":")[1].split(" - ")[1].split("=")[0])
				f.close()
	else:
		data = []
		for i in range(len(paramvalsarray)):
			data.append(float(oflines[i]))
		data = np.array(data)
		data = data.reshape((len(paramvalsarray), 1))
	return data

class MadsYaml:
	def __init__(self, filename, data=[]):
		self.filename = filename
		self.filenameroot = ".".join(filename.split(".")[:-1])
		if data == []:
			if os.path.exists ( filename ):
				stream = file(filename, 'r')
				# self.data = UnsortableOrderedDict(yaml.load(stream)) # non-sorting does not work
				self.data = yaml.load(stream)
				stream.close()
			else:
				print 'WARNING: MADS input file ' + filename + ' is missing'
		else:
			self.data = data
		self.obsIndices = {}
		for i in range(len(self.data["Observations"])):
			self.obsIndices[self.data["Observations"][i].keys()[0]] = i
		self.paramIndices = {}
		for i in range(len(self.data["Parameters"])):
			self.paramIndices[self.data["Parameters"][i].keys()[0]] = i
	def dump(self, filename, data=[], forceproblemtop=False):
		stream = file(filename, 'w')
		if data == []:
			data = self.data
		if forceproblemtop:
			#mads needs the problem keyword to be first, so we make that happen using the trick below
			copydata = {"Problem": data["Problem"]}
			yaml.dump(copydata, stream, width=255)
			copydata = dict(data)
			del copydata["Problem"]
			yaml.dump(copydata, stream, width=255)
		else:
			yaml.dump(data, stream, width=255)
		stream.close()
	def dumpcsv(self, filename):
		f = open(filename, "w")
		f.write("x,y,z0,z1,t,c\n")
		for well in self.data['Wells']:
			well_key = well.keys()[0]
			x = well[well_key]['x']
			y = well[well_key]['y']
			z0 = well[well_key]['z0']
			z1 = well[well_key]['z1']
			for obs in well[well_key]['obs']:
				t = obs[obs.keys()[0]]['t']
				c = obs[obs.keys()[0]]['c']
				if obs[obs.keys()[0]]['weight'] >= 0:
					f.write(",".join(map(str, [x, y, z0, z1, t, c])) + "\n")
		f.close()
	def deleteProblemFlag(self, flagName):
		try:
			del self.data["Problem"][flagName]
		except KeyError:
			pass #this is fine -- maybe the key just wasn't in there
	def setProblemFlag(self, flagName, typeval):
		self.data["Problem"][flagName] = typeval
	def setParamAttribute(self, paramName, paramAttribute, typeval, op=int):
		try:
			i = self.paramIndices[paramName]
			self.data["Parameters"][i][paramName][paramAttribute] = op(str(typeval))
			return
		except KeyError:
			#if we are here, that means the param was not in "Parameters" -- it must be in Sources[0]!!! (or not, but, being fools, we assume that it is there)
			sourcekey = self.data["Sources"][0].keys()[0]
			self.data["Sources"][0][sourcekey][paramName][paramAttribute] = op(str(typeval))
	def getParamKeyword(self, paramName, keyword):
		i = self.paramIndices[paramName]
		return self.data["Parameters"][i][paramName][keyword]
	def getObsKeyword(self, obsName, keyword):
		i = self.obsIndices[obsName]
		return self.data["Observations"][i][obsName][keyword]
	#TODO i will suggest to replace setParamMax, ..., etc with setParamAttribute (above)
	def setParamMax(self, paramName, maxval):
		setParamAttribute(self, paramName, "max", maxval, op=float)
		return
	def setParamMin(self, paramName, minval):
		setParamAttribute(self, paramName, "min", minval, op=float)
		return
	def setParamType(self, paramName, typeval):
		setParamAttribute(self, paramName, "type", typeval, op=str)
		return
	def setParamStep(self, paramName, stepval):
		setParamAttribute(self, paramName, "step", stepval, op=float)
		return
	def setParam(self, paramName, value):
		setParamAttribute(self, paramName, "init", value, op=float)
		return
	def getWellLocations(self):
		xw = []
		yw = []
		for well in self.data["Wells"]:
			wellkey = well.keys()[0]
			xw.append(well[wellkey]["x"])
			yw.append(well[wellkey]["y"])
		return xw, yw
	def getWellNames(self):
		nw = []
		for well in self.data["Wells"]:
			wellkey = well.keys()[0]
			nw.append(wellkey)
		return nw
	def getWell(self, wellname):
		for well in self.data["Wells"]:
			wellkey = well.keys()[0]
			if wellkey == wellname:
				return well[wellkey]
		raise KeyError
	def getNumObs(self):
		numobs = 0
		for well in self.data["Wells"]:
			wellkey = well.keys()[0]
			numobs += len(well[wellkey]['obs'])
		return numobs
	def getNumSources(self):
		return len(self.data["Sources"])
	def getSourceType(self, i):
		return self.data["Sources"][i].keys()[0]
	def getSourceParams(self, i):
		sourcetype = self.getSourceType(i)
		return self.data["Sources"][i][sourcetype]
	def getParamAttribute(self, paramName, attributeName):
		i = self.paramIndices[paramName]
		return self.data["Parameters"][i][paramName][attributeName]
	def runSaltelli(self, xyzt, num_processes, num_realizations, working_dir):
		support_data = map(lambda position: [self, position, num_realizations, working_dir], xyzt)
		#send data to the multiprocessing queue
		pool = multiprocessing.Pool(num_processes)
		pool.map(run_saltelli_support_func, support_data)
	def runSaltelliAllObs(self, num_process, num_realizations, working_dir):
		xyzt = []
		for well in self.data['Wells']:
			well_key = well.keys()[0]
			well[well_key]['weight'] = 1#this makes sure we only do SA on one well at a time
			x = well[well_key]['x']
			y = well[well_key]['y']
			z = well[well_key]['z0']
			for obs in well[well_key]['obs']:
				t = obs[obs.keys()[0]]['t']
				xyzt.append((x, y, z, t))
		return self.runSaltelli(xyzt, num_process, num_realizations, working_dir)
	#resets the observations at each well
	def setObs(self, times, concentrations=[], weights=[], logs=[], mins=[], maxes=[]):
		if len(concentrations) != len(times):
			concentrations = map(lambda x: 0, times)
		if len(weights) != len(times):
			weights = map(lambda x: 1, times)
		if len(logs) != len(times):
			logs = map(lambda x: 0, times)
		if len(mins) != len(times):
			mins = map(lambda x: 0, times)
		if len(maxes) != len(times):
			maxes = map(lambda x: 1e6, times)
		obs = map(lambda i, t, c, w, l, mi, ma: {i: {'t': t, 'c': c, 'weight': w, 'log': l, 'min': mi, 'max': ma }}, range(1, len(times) + 1), times, concentrations, weights, logs, mins, maxes)
		for well in self.data['Wells']:
			well_key = well.keys()[0]
			well[well_key]['obs'] = copy.deepcopy(obs)
	def setWellsObsAttribute(self, attribute, val):
		for well in self.data['Wells']:
			well_key = well.keys()[0]
			for obs in well[well_key]['obs']:
				obs[obs.keys()[0]][attribute] = val
	#TODO we may want to replace below with above ...
	def setAlpha(self, alpha):
		for well in self.data['Wells']:
			well_key = well.keys()[0]
			for obs in well[well_key]['obs']:
				obs[obs.keys()[0]]['alpha'] = alpha
	def setScale(self, scale):
		for well in self.data['Wells']:
			well_key = well.keys()[0]
			for obs in well[well_key]['obs']:
				obs[obs.keys()[0]]['scale'] = scale
	def horizonOfUncertaintyToBounds(self, horizon):
		alpha_low = .5 + 1.5 / (1 + horizon)
		alpha_high = 2.
		scale_factor_low = 2 ** -horizon
		scale_factor_high = 2 ** horizon
		return ([alpha_low, scale_factor_low], [alpha_high, scale_factor_high])
	def bayesianInfoGapInnerMaximum(self, horizon, num_processes):
		lower_bounds, upper_bounds = self.horizonOfUncertaintyToBounds(horizon)
		inner_max = ance_min(bayesianProbabilityOfFailure, lower_bounds, upper_bounds, 200, 3, num_processes)
		return inner_max
	def bayesianProbabilityOfFailure(self, alpha, scale_factor, working_dir, num_samples, horizon_of_uncertainty):
		copy_data = copy.deepcopy(self.data)
		for well in copy_data['Wells']:
			well_key = well.keys()[0]
			for obs in well[well_key]['obs']:
				if obs[obs.keys()[0]]['weight'] >= 0:
					scale = obs[obs.keys()[0]]['scale']
					obs[obs.keys()[0]]['scale'] = scale * scale_factor
					obs[obs.keys()[0]]['alpha'] = alpha
				else:
					obs[obs.keys()[0]]['max'] /= (1 + horizon_of_uncertainty)
		short_filename = get_short_filename(self.filename)
		long_filename = working_dir + '/' + short_filename + '-alpha' + str(alpha) + '-factor' + str(scale_factor) + '-hou' + str(horizon_of_uncertainty) + '.mads'
		self.dump(long_filename, data=copy_data)
		#call_string = mads_bin + ' ' + long_filename + ' bayes nosin real=' + str(num_samples) + ' 2>/dev/null | grep \"probability of failure: \"'
		call_string = mads_bin + ' ' + long_filename + ' bayes nosin real=' + str(num_samples) + ' | grep \"probability of failure: \"'
		#call_string = mads_bin + ' ' + long_filename + ' bayes nosin real=' + str(num_samples)
		try:
			output = subprocess.check_output(call_string, shell=True)
		except:
			print "epic fail on call_string: " + call_string
			sys.exit(1)
		pfail = float(output.split()[3])
		nsamples = int(output.split()[4].split('(')[1])
		return (pfail, nsamples)
	def parIGRND(self, num_procs, num_runs_per_proc, dir_root="run"):
		for i in range(num_procs):
			dir_name = dir_root + str(i)
			subprocess.call("mkdir " + dir_name, shell=True)
			self.dump(dir_name + "/" + self.filename)
		pool = multiprocessing.Pool(num_procs)
		seed = random.randint(0, 2 ** 30)
		pool.map(mads_exec, map(lambda i: dir_root + str(i) + "/" + self.filename + " igrnd real=" + str(num_runs_per_proc) + " seed=" + str(seed + i), range(num_procs)))
	def getIGRNDreruns(self, num_procs, dir_root="run"):
		pool = multiprocessing.Pool(num_procs)
		#out restart=out.igrnd.results forward success save
		pool.map(mads_exec, map(lambda i: dir_root + str(i) + "/" + self.filename + " resultsfile=" + dir_root + str(i) + "/" + self.filenameroot + ".igrnd.results forward success save", range(num_procs)))
	def plotIGRND(self, root_filename, num_sources, dir_root=[], num_source_params=6, velocity_factor=1, dispersion_factor=1, num_procs=[]):
		#this code assumes that the parameters are listed in the order (x,y,dx,dy,...)^num_sources,theta,v,ax,ay,az...
		x = []
		dx = []
		y = []
		dy = []
		theta = []
		v = []
		ax = []
		ay = []
		if dir_root == []:
			residual_filenames = [root_filename + ".igrnd.results"]
		elif num_procs == []:
			residual_filenames = [dir_root + "/" + root_filename + ".igrnd.results"]
		else:
			residual_filenames = []
			for i in range(num_procs):
				residual_filenames.append(dir_root + str(i) + "/" + root_filename + ".igrnd.results")
		for filename in residual_filenames:
			with open(filename, "r") as f:
				lines = f.readlines()
				f.close()
			for line in lines:
				if "success 1" in line:
					final_params_str = line.split(":")[3]
					final_param_vals = map(float, final_params_str.split()[2:])
					for i in range(num_sources):
						 x.append(final_param_vals[num_source_params * i])
						 y.append(final_param_vals[num_source_params * i + 1])
						 dx.append(final_param_vals[num_source_params * i + 2])
						 dy.append(final_param_vals[num_source_params * i + 3])
					theta.append(final_param_vals[num_source_params * num_sources])
					v.append(final_param_vals[num_source_params * num_sources + 1])
					ax.append(final_param_vals[num_source_params * num_sources + 2])
					ay.append(final_param_vals[num_source_params * num_sources + 3])
		for i in range(len(x)):
			rect = plt.Rectangle((x[i] - .5 * dx[i], y[i] - .5 * dy[i]), dx[i], dy[i], fc=(1., 0, 0, 16. / (len(x) * num_sources)), ec=(0, 0, 0, 0))
			plt.gca().add_patch(rect)
		for i in range(len(theta)):
			points = [[0, 0], [velocity_factor * v[i] * np.cos(theta[i] * np.pi / 180.), velocity_factor * v[i] * np.sin(theta[i] * np.pi / 180.)]]
			line = plt.Polygon(points, ec=(0., 0., 0., 4. / len(theta)), fill=None, closed=None)
			plt.gca().add_patch(line)
			ellipse = matplotlib.patches.Ellipse((2000, -250), dispersion_factor * v[i] * ax[i], dispersion_factor * v[i] * ay[i], ec=(0, 0, 0, 0), fc=(0., 0., 1., 1. / len(theta)))
			plt.gca().add_patch(ellipse)
		for well in self.data["Wells"]:
			well_key = well.keys()[0]
			plt.annotate(well_key, xy=(well[well_key]["x"], well[well_key]["y"]), xytext=(0, 0), textcoords="offset points")
		plt.axis('scaled')
		plt.show()

def run_saltelli_support_func(support_data):
	my = support_data[0]
	position = support_data[1]
	num_realizations = support_data[2]
	working_dir = support_data[3]
	copy_data = copy.deepcopy(my.data)
	#modify copy_data
	well_key = copy_data['Wells'][0].keys()[0]
	copy_data['Wells'][0][well_key]['x'] = position[0]
	copy_data['Wells'][0][well_key]['y'] = position[1]
	copy_data['Wells'][0][well_key]['z0'] = position[2]
	copy_data['Wells'][0][well_key]['z1'] = position[2]
	copy_data['Wells'][0][well_key]['obs'][0][1]['t'] = position[3]
	copy_data['Wells'][0][well_key]['obs'][0][1]['weight'] = -1
	#dump to file
	short_filename = get_short_filename(my.filename)
	long_filename = working_dir + '/' + short_filename + '-x' + str(position[0]) + '-y' + str(position[1]) + '-z' + str(position[2]) + '-t' + str(position[3]) + '.mads'
	stream = file(long_filename, 'w')
	yaml.dump(copy_data, stream, width=255) #TODO replace with copy_data.dump(stream)
	#setup the strings that will be used to call mads on each file
	call_string = mads_bin + ' ' + long_filename + ' gsens salt nosin real=' + str(num_realizations) + ' >/dev/null'
	subprocess.call(call_string, shell=True)

def get_short_filename(filename):
	pos = filename.find('.mads')
	if pos < 0: # not found
		short_filename = filename
	else:
		short_filename = filename[0:pos] #get rid of the .mads
	return short_filename

def runBayesIG(filename='bt3/bt3.mads', working_dir='bt3/results', num_samples=1e4, num_procs=1, hakuna_matata=1, discretization_factor=5, i0=0):
	my = MadsYaml(filename)
	pool = multiprocessing.Pool(num_procs)
	min_alpha = lambda h: .5 + 1.5 / (1. + h / 10.)
	max_factor = lambda h: 2 ** h
	min_factor = lambda h: .5 ** h
	"""
	alpha_list = [min_alpha(0)]
	factor_list = [max_factor(0)]
	h_list = [0]
	"""
	alpha_list = []
	factor_list = []
	h_list = []
	for i in range(i0, hakuna_matata * discretization_factor + 1):
		for j in range(i + 1):
			for k in range(i + 1):
				alpha_list.append(min_alpha(j / float(discretization_factor)))
				factor_list.append(max_factor(k / float(discretization_factor)))
				h_list.append(i / float(discretization_factor))
				if k != 0:
					alpha_list.append(min_alpha(j / float(discretization_factor)))
					factor_list.append(min_factor(k / float(discretization_factor)))
					h_list.append(i / float(discretization_factor))
	"""
	for x in  zip(alpha_list, factor_list, h_list):
		print x
	sys.exit(0)
	"""
	"""
	for i in range(hakuna_matata * discretization_factor + 1):
		for k in range(0, i):
			alpha_list.append(min_alpha(i / float(discretization_factor)))
			factor_list.append(max_factor(k / float(discretization_factor)))
			h_list.append(i / float(discretization_factor))
		for k in range(0, i + 1):
			alpha_list.append(min_alpha(k / float(discretization_factor)))
			factor_list.append(max_factor(i / float(discretization_factor)))
			h_list.append(i / float(discretization_factor))
			"""
	madsyaml_list = [my for i in range(0, len(h_list))]
	workingdir_list = [working_dir for i in range(0, len(h_list))]
	numsamples_list = [num_samples for i in range(0, len(h_list))]
	eval_list = []
	i = 0
	while all(map(lambda x: x[0] < 0.1, eval_list)) and i < len(h_list):
		n = min(i + num_procs, len(h_list))
		short_madsyaml_list = madsyaml_list[i:n]
		short_h_list = h_list[i:n]
		short_alpha_list = alpha_list[i:n]
		short_factor_list = factor_list[i:n]
		short_workingdir_list = workingdir_list[i:n]
		short_numsamples_list = numsamples_list[i:n]
		zip_list = zip(short_madsyaml_list, short_alpha_list, short_factor_list, short_workingdir_list, short_numsamples_list, short_h_list)
		new_evals = pool.map(bayesian_probability_of_failure_support, zip_list)
		eval_list.extend(new_evals)
		for j in range(0, len(new_evals)):
			print (h_list[i + j], eval_list[i + j], alpha_list[i + j], factor_list[i + j])
		i += num_procs

def bayesian_probability_of_failure_support(zipped):
	my = zipped[0]
	alpha = zipped[1]
	factor = zipped[2]
	working_dir = zipped[3]
	num_samples = zipped[4]
	h = zipped[5]
	return my.bayesianProbabilityOfFailure(alpha, factor, working_dir, num_samples, h)

def testSaltelli():
	xyzt = [(100,50,1,10), (200, 100, 1, 20), (300, 150, 1, 10), (400, 200, 1, 20)]
	#xyzt = [(100,50,1,10), (200, 100, 1, 20), (300, 150, 1, 10), (400, 200, 1, 20), (100,50,1,11), (200, 100, 1, 21), (300, 150, 1, 11), (400, 200, 1, 21)]
	#xyzt = [(100,50,1,10), (200, 100, 1, 20), (300, 150, 1, 10), (400, 200, 1, 20), (100,50,1,11), (200, 100, 1, 21), (300, 150, 1, 11), (400, 200, 1, 21), (101,50,1,10), (201, 100, 1, 20), (301, 150, 1, 10), (401, 200, 1, 20), (101,50,1,11), (201, 100, 1, 21), (301, 150, 1, 11), (401, 200, 1, 21), (102,50,1,10), (202, 100, 1, 20), (302, 150, 1, 10), (402, 200, 1, 20), (102,50,1,11), (202, 100, 1, 21), (302, 150, 1, 11), (402, 200, 1, 21), (103,50,1,10), (203, 100, 1, 20), (303, 150, 1, 10), (403, 200, 1, 20), (103,50,1,11), (203, 100, 1, 21), (303, 150, 1, 11), (403, 200, 1, 21)]
	#my = MadsYaml('levy2.mads')
	my = MadsYaml('a01.mads')
	#print len(xyzt)
	my.runSaltelli(xyzt, 8, 10000, 'test')
	#my.runSaltelliAllObs(4, 10000, 'test3')

def mads_exec(cmdline): #TODO this should be used for all MADS executions
	call_string = mads_bin + ' ' + cmdline
	#print "MADS: Execute: " + call_string
	try:
		output = subprocess.check_output(call_string, shell=True)
	except:
		print "MADS: Execution failed: " + call_string
		sys.exit(1)

def mads_create_truth(prob):
	mads_exec(prob + " create")

def mads_copy_wells(a,b,c):
	m1 = MadsYaml(a)
	m2 = MadsYaml(b)
	m2.data['Wells'] = m1.data['Wells']
	m2.data['Problem'] = "" # Blank problem setup
	m2.dump(c)

def cb(a,b):
	root_name_split = a.split(".")
	root_name = root_name_split[0]
	print "Root:", root_name
	c = 'cb.mads'
	print "MADS: copy wells structure from " + a + " in " + b + " and save it in " + c
	mads_copy_wells(a,b,c)
	mads_create_truth(c)
	print "MADS: Create known-source (sk) problem: " + root_name + '-sk.mads'
	subprocess.check_output('cp cb-truth.mads ' + root_name + '-sk.mads', shell=True) # Create known-source (sk) problem
	mads_copy_wells('cb-truth.mads',a,'cb-test.mads')
	mads_exec( 'cb-test.mads forward' ) # Test and reformat
	print "MADS: Create unknown-source (sk) problem: " + root_name + '-su.mads'
	subprocess.check_output('mv cb-test-rerun.mads ' + root_name + '-su.mads', shell=True) # Create unknown-source (su) problem
	subprocess.check_output('rm cb*', shell=True) # Clean

"""
if __name__ == "__main__":
	print 'Number of arguments:', len(sys.argv)
	print 'Arguments:', str(sys.argv)
	my = MadsYaml(sys.argv[1:])
	my.runSaltelliAllObs(4, 10000, 'test2')
	"""

