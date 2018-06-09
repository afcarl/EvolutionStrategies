import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np

ALL_PERTUBATIONS = []

def normalize_dico(dico):

	normalizer = nn.Softmax(dim = 0)
	values = list(dico.values())

	normalized_values = normalizer(torch.tensor(values).float().reshape(-1)).detach().numpy()
	new_dico = {}
	for i,s in enumerate(dico): 
		new_dico[s] = normalized_values[i]

	return new_dico

class Ind(nn.Module): # Basic individual class 

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

class Swarm(): # Population class


	def __init__(self):

		self.base = Ind()
		self.fitness = {}
		# self.seeds = []

		self.std = 0.3
		self.lr = 1e-2

	def add_pop(self, no_perturbation = False): # This creates a clone, for evaluating the gradient 

		seed = np.random.randint(350000)
		print(seed)
		torch.manual_seed(seed)
		clone = self.base.get_clone()
		if no_perturbation: 
			return clone 

		for p in clone.state_dict().values(): 

			perturbation = torch.ones_like(p).normal_()
			ALL_PERTUBATIONS.append(perturbation)

			p.copy_(p + perturbation*self.std)

		# self.seeds.append(seed)
		return clone, seed

	def observe_fitness(self, seed, fitness): 

		self.fitness[seed] = fitness

	def improve(self):  # REINFORCE part 

		normalized_fitness = normalize_dico(self.fitness) # loss is normalized, using softmax 
		for p in self.base.state_dict().values(): 
			grads = torch.zeros_like(p)
			for s in normalized_fitness: 
				print(s)
				torch.manual_seed(s)
				perturbation = torch.ones_like(p).normal_()
				print(perturbation)
				merit = normalized_fitness[s].astype(float)
				movement = perturbation*torch.tensor(merit)#.float().expand_as(perturbation)
				# input(movement.shape)
				grads += movement

			p.copy_(p.data + grads*self.lr/(self.std*len(self.fitness)))

		self.fitness = {}

	def improve_bis(self): # Other way to try to recover the perturbation 

		normalized_fitness = normalize_dico(self.fitness)
		grads = [torch.zeros_like(p) for p in self.base.parameters()]
		for i,s in enumerate(normalized_fitness): 
			print(s)
			torch.manual_seed(s)
			for param, v in enumerate(self.base.state_dict().values()) : 
				perturbation = torch.ones_like(v).normal_()
				print(perturbation)
				grads[param] += perturbation*normalized_fitness[s].astype(float)

		for g,v in zip(grads,self.base.state_dict().values()): 
			v.copy_(v+g*self.lr/(self.std*len(self.fitness)))
		
		self.fitness = {}

s = Swarm()
i1, seed = s.add_pop()
for p in ALL_PERTUBATIONS: 
	print(p)

s.observe_fitness(seed,1)
s.improve_bis()
input()


input('Starting...')


 #--------------- XOR Dataset

data = torch.tensor(
	[[0,1],
	 [1,0], 
	 [0,0], 
	 [1,1]]).float()
data_y = torch.tensor([1,1,0,0]).float().reshape(-1,1)


epochs = 1000
pop = 1

for epoch in range(epochs): 

	overall_loss = 0. 
	for _ in range(pop): 
		ind, seed = s.add_pop()
		pred = ind(data)
		loss = F.mse_loss(pred, data_y) 
		s.observe_fitness(seed, -loss.item()) # Minus loss as performance 

		overall_loss += loss.item()
	s.improve()
	print('Epoch {} -- Loss {}'.format(epoch, overall_loss/pop))


