import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np


def normalize_dico(dico):

	normalizer = nn.Softmax(dim = 0)
	values = list(dico.values())

	normalized_values = normalizer(torch.tensor(values).float().reshape(-1)).detach().numpy()
	new_dico = {}
	for i,s in enumerate(dico): 
		new_dico[s] = normalized_values[i]

	return new_dico

class Ind(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)
		self.l1 = nn.Linear(2, 5)
		self.l2 = nn.Linear(5,1)

	def forward(self,x): 

		x = F.relu(self.l1(x))
		out = F.relu(self.l2(x))

		return out 

	def get_clone(self): 

		clone = Ind()
		for clone_p, source_p in zip(clone.state_dict().values(), self.state_dict().values()): 
			clone_p.copy_(source_p)

		return clone 

class Swarm(): 


	def __init__(self):

		self.base = Ind()
		self.fitness = {}
		# self.seeds = []

		self.std = 0.3
		self.lr = 1e-2

	def add_pop(self, no_perturbation = False):

		seed = np.random.randint(350000)
		torch.manual_seed(seed)

		clone = self.base.get_clone()
		if no_perturbation: 
			return clone 

		for p in clone.state_dict().values(): 

			perturbation = torch.ones_like(p).normal_()*self.std
			p.copy_(p + perturbation)

		# self.seeds.append(seed)
		return clone, seed

	def observe_fitness(self, seed, fitness): 

		self.fitness[seed] = fitness

	def improve(self): 

		normalized_fitness = normalize_dico(self.fitness)
		for p in self.base.state_dict().values(): 
			grads = torch.zeros_like(p)
			for s in normalized_fitness: 
				torch.manual_seed(s)
				perturbation = torch.ones_like(p).normal_()
				merit = normalized_fitness[s].astype(float)
				movement = perturbation*torch.tensor(merit)#.float().expand_as(perturbation)
				# input(movement.shape)
				grads += movement

			p.copy_(p.data + grads*self.lr/(self.std*len(self.fitness)))

		self.fitness = {}
		

s = Swarm()

data = torch.tensor(
	[[0,1],
	 [1,0], 
	 [0,0], 
	 [1,1]]).float()
data_y = torch.tensor([1,1,0,0]).float().reshape(-1,1)


epochs = 1000
pop = 20

for epoch in range(epochs): 

	overall_loss = 0. 
	for _ in range(pop): 
		ind, seed = s.add_pop()
		pred = ind(data)
		loss = F.mse_loss(pred, data_y)
		s.observe_fitness(seed, -loss.item())

		overall_loss += loss.item()
	s.improve()
	print('Epoch {} -- Loss {}'.format(epoch, overall_loss/pop))


