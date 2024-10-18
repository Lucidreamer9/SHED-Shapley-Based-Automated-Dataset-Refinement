# SHED-Shapley-Based-Automated-Dataset-Refinement

paper link: https://arxiv.org/abs/2405.00705

Before running the project, please make sure that you have installed the required dependencies.

Please prepare your original dataset and make sure it's in the right format.
The scripts in the project have been integrated into the run.sh script, which handles the entire workflow. Ensure that all file paths are correct. The final selected dataset will be saved in final_dataset.

The base model we use is llama-7b, ensure that you have sufficient computational resources to run the models, especially during the Shapley value calculation and model fine-tuning steps.
The path and format of the original dataset must comply with the scripts' requirements.
