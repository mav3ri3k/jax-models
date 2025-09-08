import wandb
api = wandb.Api()

run = api.run("mav3ri3k-vellore-institute-of-technology/chess/78fvobev")
history = run.scan_history(keys=["Test_loss"])
losses = [row["Test_loss"] for row in history]
print(losses)
