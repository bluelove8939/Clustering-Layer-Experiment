import Resnet50_pruned_STL10_normal as normal
from tools.pruning import PruneModule


model = normal.model
tuning_dataloader = normal.train_loader
loss_fn = normal.loss_fn
optimizer = normal.optimizer

pmodule = PruneModule(tuning_dataloader=tuning_dataloader, loss_fn=loss_fn, optimizer=optimizer)
pmodule.prune_model(model, target_amount=0.3, threshold=1, step=0.1, max_iter=5, pass_normal=False)