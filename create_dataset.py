from test_maneuvers import test_maneuvers
import random

min_targets = 1
max_targets = 5

train_samples = 1000
test_samples = 100

# First generating training samples

for i in range(train_samples):
    num_targets = random.randint(min_targets,max_targets)
    test_maneuvers(i, num_targets, type="Train")
    
    
# Now generating test samples

for i in range(test_samples):
    num_targets = random.randint(min_targets,max_targets)
    test_maneuvers(i, num_targets, type="Test")
    
print("Dataset Generation Complete")