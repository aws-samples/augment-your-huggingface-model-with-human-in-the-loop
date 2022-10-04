# Augment your HuggingFace model with Human-in-the-Loop

This repository is part of a workshop that was originally developed to be delivered at the [NLP London Summit](https://www.aws-nlp-summit.com/). 

<br>

<h3>Problem Statement</h3>

The purpose of this example is to showcase how you can integrate Human-in-the-Loop in your typical Machine Learning workflow.
In this example, we assume the need for a sentiment analysis model that when used to make inferences we may require a human to review the prediction, label it based on their own view & understanding of the sentiment and finally, when the model is re-trained (on-demand) we expect it to be fine tuned utilising any new human-annotated examples that were created in the meantime. This way we are creating an ever improving model that everytime it is unsure of a prediction, a human is adding one more datapoint that the model will use to improve at the next iteration. 


<br>

<h3>Workshop Structure</h3>

The structure of the repository attempts to be as simple as possible and is structured with numbered notebooks and includes self-explanatory comments allowing this code to be able to executed unatended. 

1. Start by running notebook `1_create_model_pipeline` which will create a SageMaker Pipeline that Builds, Trains and Registers in Model Registry a HuggingFace model. When first run, it will take a few minutes to complete.
2. While the above is running, navigate to the notebook `2A_a2i_setup` which will take you through the process of preparing the A2I (Augmented AI) service that will help us implement the Human-in-the-loop workflows.
3. Next, and once the execution of all steps in notebook 1 have succedded, you may open notebook `2B_deploy_model` that will guide you in deploying the latest model in a simple real-time endpoint. This should take no more than 5 minutes to finish
4. Forth step and notebook to use is `3_generate_some_traffic`. Here we are utilising the endpoint we created at step 3 and the resources created at step 2 to pass some artificial traffic to the endpoint and have a portion of the traffic (the cases where the sentiment prediction is not very strong) to be sent for human labelling. <i>Please note, that depending on your use-case, this step would probably be performed on the application layer that is utilising the endpoint. Here we are using a notebook for demonstration purposes only. </i>
5. Once you have generated some traffic, you need to navigate to the reviewer panel and as a Human Labeller, annotate the sentences that were sent for human annotation at step 4
6. Finally, you may want to kick off another training cycle. Here we trigger a new run of the original pipeline, that will automatically pick up any new annotations available and use these to further tune our original HuggingFace Model. 

<br>

<h3>Workshop Execution Notes</h3>
<li> If running this in your own account, make sure to delete the deployed endpoint at the end of the workshop as this will incur costs if left running. When you have finished with the workshop navigate to notebook `2B_deploy_model.ipynb`, uncomment that last cell and run the command to delete the endpoint. 
<li> When openning any of the notebooks withing SageMaker Studio, you will be opted to choose a kernel. Choose `Data Science` (although almost every one of the availbe ones would work)
</li>
    
<br><br>


<h3>Technical notes:</h3>
<li> The base model used in ths workshop is a distilbert-base-uncased and fine tuned on small portion of imdb data.
<li> Only a small portion of training data are used since this is a live workshop and we need to strike balance between performance of any given model and the developer experience and time needed to complete this.
<li> In this example the retraining of the training pipeline is done "manually" or via code. In your scenario this can be automated based on your needs (on a schedule, every time there is a new annotation etc.)
<li> Similarly deployment of model can be automated when a model is approved but it is skipped in this workshop for simlicity. 
<li> The use of the retry mechanism for the training step is optional and is there to show you this capability of SageMaker Pipelines. In the context of the instructor led workshop, the demo accounts provided might not have access to the GPU instance required right away and consecutive/concurrent executions will exceed the account limits for the `p3.2xlarge` instance. The retry mechanism will add the necessary wait time until these resources are available.
</li>

<br>
<br>
    
If you found this workshop valueable or you have any comments, questions or observations, please reach out!
    
## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

----

Author: Georgios Schinas