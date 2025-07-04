import wandb

# Initialize a run
run = wandb.init(project="capsule_vision_challenge_2024")

# Create an artifact
artifact = wandb.Artifact(name="anat_1_yolo_split", type="dataset")
artifact.add_dir("/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_yolo_split")  # or artifact.add_file("path/to/file")

# Log the artifact
run.log_artifact(artifact)
run.finish()

#TO ADD VERSIONS
# import wandb
# wandb.init(entity='lakshminarayanan-', project='capsule_vision_challenge_2024')
# art = wandb.Artifact('anat_1_yolo_split', type='dataset')
# # ... add content to artifact ...
# wandb.log_artifact(art)

#DELETE
# import wandb
# api = wandb.Api()

# # Delete run
# run = api.run("lakshminarayanan-/capsule_vision_challenge_2024/6finr72o")
# run.delete()

# # Delete artifact
# artifact = api.artifact("lakshminarayanan-/capsule_vision_challenge_2024/anat_1_yolo_split:v0")
# artifact.delete()
