# Text-Summarization-using-Transformers-T5


### Introduction

we will  fine tuning a transformer model for **Summarization Task**. 
In this task a summary of a given article/document is generated when passed through a network. There are 2 types of summary generation mechanisms:

1. ***Extractive Summary:*** the network calculates the most important sentences from the article and gets them together to provide the most meaningful information from the article.
2. ***Abstractive Summary***: The network creates new sentences to encapsulate maximum gist of the article and generates that as output. The sentences in the summary may or may not be contained in the article. 

we will be generating ***Abstractive Summary***. 


* We will be using : [Weights and Biases Service](https://www.wandb.com/) WandB in short.
* It is a experiment tracking, parameter optimization and artifact management service. That can be very easily integrated to any of the Deep learning or Machine learning frameworks. 

The notebook will be divided into separate sections to provide a organized walk through for the process used. This process can be modified for individual use cases. The sections are:

1. [Preparing Environment and Importing Libraries](#section01)
2. [Preparing the Dataset for data processing: Class](#section02)
3. [Fine Tuning the Model: Function](#section03)
4. [Validating the Model Performance: Function](#section04)
5. [Main Function](#section05)
    * [Initializing WandB](#section501)
    * [Importing and Pre-Processing the domain data](#section502)
    * [Creation of Dataset and Dataloader](#section503)
    * [Neural Network and Optimizer](#section504)
    * [Training Model and Logging to WandB](#section505)
    * [Validation and generation of Summary](#section506)
6. [Examples of the Summary Generated from the model](#section06)


#### Technical Details

This script leverages on multiple tools designed by other teams. Details of the tools used below. Please ensure that these elements are present in your setup to successfully implement this script.

- **Data**:
	- We are using the News Summary dataset available at [Kaggle](https://www.kaggle.com/sunnysai12345/news-summary)
	- This dataset is the collection created from Newspapers published in India, extracting, details that are listed below.  We are referring only to the first csv file from the data dump: `news_summary.csv`
	- There are`4514` rows of data.  Where each row has the following data-point:
		- **author** : Author of the article
		- **date** : Date the article was published
		- **headline**: Headline for the published article
		- **read_more** : URL for the article to follow online
		- **text**: This is the summary of the article
		- **ctext**: This is the complete article


- **Language Model Used**: 
    - This notebook uses one of the most recent and novel transformers model ***T5***. [Research Paper](https://arxiv.org/abs/1910.10683)    
    - ***T5*** in many ways is one of its kind transformers architecture that not only gives state of the art results in many NLP tasks, but also has a very radical approach to NLP tasks.
    - **Text-2-Text** - According to the graphic taken from the T5 paper. All NLP tasks are converted to a **text-to-text** problem. Tasks such as translation, classification, summarization and question answering, all of them are treated as a text-to-text conversion problem, rather than seen as separate unique problem statements.
    - **Unified approach for NLP Deep Learning** - Since the task is reflected purely in the text input and output, you can use the same model, objective, training procedure, and decoding process to ANY task. Above framework can be used for any task - show Q&A, summarization, etc. 
   - We will be taking inputs from the T5 paper to prepare our dataset prior to fine tuning and training.    
   - [Documentation for python](https://huggingface.co/transformers/model_doc/t5.html)




- Hardware Requirements: 
	- Python 3.6 and above
	- Pytorch, Transformers and
	- All the stock Python ML Library
	- GPU enabled setup 
   

- **Script Objective**:
	- The objective of this script is to fine tune ***T5 *** to be able to generate summary, that a close to or better than the actual summary  while ensuring the important information from the article is not lost.

---


<a id='section02'></a>
### Preparing the Dataset for data processing: Class

We will start with creation of Dataset class - This defines how the text is pre-processed before sending it to the neural network. This dataset will be used the the Dataloader method that will feed  the data in batches to the neural network for suitable training and processing. 
The Dataloader and Dataset will be used inside the `main()`.
Dataset and Dataloader are constructs of the PyTorch library for defining and controlling the data pre-processing and its passage to neural network. For further reading into Dataset and Dataloader .

#### *CustomDataset* Dataset Class
- This class is defined to accept the Dataframe as input and generate tokenized output that is used by the **T5** model for training. 
- We are using the **T5** tokenizer to tokenize the data in the `text` and `ctext` column of the dataframe. 
- The tokenizer uses the ` batch_encode_plus` method to perform tokenization and generate the necessary outputs, namely: `source_id`, `source_mask` from the actual text and `target_id` and `target_mask` from the summary text.
- The *CustomDataset* class is used to create 2 datasets, for training and for validation.
- *Training Dataset* is used to fine tune the model: **80% of the original data**
- *Validation Dataset* is used to evaluate the performance of the model. The model has not seen this data during training. 

#### Dataloader: Called inside the `main()`
- Dataloader is used to for creating training and validation dataloader that load data to the neural network in a defined manner. This is needed because all the data from the dataset cannot be loaded to the memory at once, hence the amount of data loaded to the memory and then passed to the neural network needs to be controlled.
- This control is achieved using the parameters such as `batch_size` and `max_len`.
- Training and Validation dataloaders are used in the training and validation part of the flow respectively


<a id='section03'></a>
### Fine Tuning the Model: Function

Here we define a training function that trains the model on the training dataset created above, specified number of times (EPOCH), An epoch defines how many times the complete data will be passed through the network. 

This function is called in the `main()`

Following events happen in this function to fine tune the neural network:
- The epoch, tokenizer, model, device details, testing_ dataloader and optimizer are passed to the `train ()` when its called from the `main()`
- The dataloader passes data to the model based on the batch size.
- `language_model_labels` are calculated from the `target_ids` also, `source_id` and `attention_mask` are extracted.
- The model outputs first element gives the loss for the forward pass. 
- Loss value is used to optimize the weights of the neurons in the network.
- After every 10 steps the loss value is logged in the wandb service. This log is then used to generate graphs for analysis.
- After every 500 steps the loss value is printed in the console.



<a id='section04'></a>
### Validating the Model Performance: Function

During the validation stage we pass the unseen data(Testing Dataset), trained model, tokenizer and device details to the function to perform the validation run. This step generates new summary for dataset that it has not seen during the training session. 

This function is called in the `main()`

This unseen data is the 20% of `news_summary.csv` which was seperated during the Dataset creation stage. 
During the validation stage the weights of the model are not updated. We use the generate method for generating new text for the summary. 

It depends on the `Beam-Search coding` method developed for sequence generation for models with LM head. 

The generated text and originally summary are decoded from tokens to text and returned to the `main()`

<a id='section05'></a>
### Main Function

The `main()` as the name suggests is the central location to execute all the functions/flows created above in the notebook. The following steps are executed in the `main()`:


<a id='section501'></a>
#### Initializing WandB 

* The `main()` begins with initializing WandB run under a specific project. This command initiates a new run for each execution of this command. 

**[WandB Service](https://www.wandb.com/)**

* This service has been created to track ML experiments, Optimize the experiments and save artifacts. It is designed to seamlessly integrate with all the Machine Learning and Deep Learning Frameworks. Each script can be organized into *Project* and each execution of the script will be registered as a *run* in the respective project.

* The service can be configured to log several default metrics, such a network weights, hardware usage, gradients and weights of the network. 

* It can also be used to log user defined metrics, such a loss in the `train()`.




* Visit the project page to see the details of different runs and what information is logged by the service. 

* Following the initialization of the WandB service we define configuration parameters that will be used across the tutorial such as `batch_size`, `epoch`, `learning_rate` etc.

* These parameters are also passed to the WandB config. The config construct with all the parameters can be optimized using the Sweep service from WandB. Currently, that is outof scope of this tutorial. 

* Next we defining seed values so that the experiment and results can be reproduced.


<a id='section502'></a>
#### Importing and Pre-Processing the domain data

We will be working with the data and preparing it for fine tuning purposes. 
*Assuming that the `news_summary.csv` is already downloaded in your `data` folder*

* The file is imported as a dataframe and give it the headers as per the documentation.
* Cleaning the file to remove the unwanted columns.
* A new string is added to the main article column `summarize: ` prior to the actual article. This is done because **T5** had similar formatting for the summarization dataset. 
* The final Dataframe will be something like this:

|text|ctext|
|--|--|
|summary-1|summarize: article 1|
|summary-2|summarize: article 2|
|summary-3|summarize: article 3|

* Top 5 rows of the dataframe are printed on the console.

<a id='section503'></a>
#### Creation of Dataset and Dataloader

* The updated dataframe is divided into 80-20 ratio for test and validation. 
* Both the data-frames are passed to the `CustomerDataset` class for tokenization of the new articles and their summaries.
* The tokenization is done using the length parameters passed to the class.
* Train and Validation parameters are defined and passed to the `pytorch Dataloader contstruct` to create `train` and `validation` data loaders.
* These dataloaders will be passed to `train()` and `validate()` respectively for training and validation action.
* The shape of datasets is printed in the console.


<a id='section504'></a>
#### Neural Network and Optimizer

* In this stage we define the model and optimizer that will be used for training and to update the weights of the network. 
* We are using the `t5-base` transformer model for our project. You can read about the `T5 model` and its features above. 
* We use the `T5ForConditionalGeneration.from_pretrained("t5-base")` commad to define our model. The `T5ForConditionalGeneration` adds a Language Model head to our `T5 model`. The Language Model head allows us to generate text based on the training of `T5 model`.
* We are using the `Adam` optimizer for our project. This has been a standard for all our tutorials and is something that can be changed updated to see how different optimizer perform with different learning rates. 
* There is also a scope for doing more with Optimizer such a decay, momentum to dynamically update the Learning rate and other parameters. All those concepts have been kept out of scope for these tutorials. 


<a id='section505'></a>
#### Training Model and Logging to WandB

* Now we log all the metrics in WandB project that we have initialized above.
* Followed by that we call the `train()` with all the necessary parameters.
* Loss at every 500th step is printed on the console.
* Loss at every 10th step is logged as Loss in the WandB service.


<a id='section506'></a>
#### Validation and generation of Summary

* After the training is completed, the validation step is initiated.
* As defined in the validation function, the model weights are not updated. We use the fine tuned model to generate new summaries based on the article text.
* An output is printed on the console giving a count of how many steps are complete after every 100th step. 
* The original summary and generated summary are converted into a list and returned to the main function. 
* Both the lists are used to create the final dataframe with 2 columns **Generated Summary** and **Actual Summary**
* The dataframe is saved as a csv file in the local drive.
* A qualitative analysis can be done with the Dataframe. 



<a id='section06'></a>
### Examples of the Summary Generated from the model

##### Example 1

**Original Text**
New Delhi, Apr 25 (PTI) Union minister Vijay Goel today batted for the unification of the three municipal corporations in the national capital saying a discussion over the issue was pertinent. The BJP leader, who was confident of a good show by his party in the MCD polls, the results of which will be declared tomorrow, said the civic bodies needed to be "revamped" in order to deliver the services to the people more effectively. The first thing needed was a discussion on the unification of the three municipal corporations and there should also be an end to the practice of sending Delhi government officials to serve in the civic bodies, said the Union Minister of State (Independent Charge) for Youth Affairs and Sports. "Barring one, the two other civic bodies have been incurring losses. It would be more fruitful and efficient if all the three were merged," he said, referring to the north, south and east Delhi municipal corporations. The erstwhile Municipal Corporation of Delhi (MCD) was trifurcated into NDMC, SDMC and EDMC by the then Sheila Dikshit-led Delhi government in 2012. Goel predicted a "thumping" victory for the BJP in the MCD polls. He said the newly-elected BJP councillors will be trained on the functioning of the civic bodies and dealing with the bureaucracy. 


**Original Summary**
Union Minister Vijay Goel has favoured unification of three MCDs ? North, South and East ? in order to deliver the services more effectively. "Barring one, the two other civic bodies have been incurring losses. It would be more fruitful and efficient if all the three were merged," he said. MCD was trifurcated into EDMC, NDMC and SDMC in 2012.

**Generated Summary**
BJP leader Vijay Goel on Saturday batted for the unification of three municipal corporations in the national capital saying a discussion over this was pertinent. "Barring one, two other civic bodies have been incurring losses," said Goels. The erstwhile Municipal Corporations of Delhi (MCD) were trifurcated into NDMC and SDMC by the then Sheilha Dikshi-led government in 2012. Notably, the MCD poll results will be declared tomorrow.




##### Example 2

**Original Text**
After much wait, the first UDAN flight took off from Shimla today after being flagged off by Prime Minister Narendra Modi.The flight will be operated by Alliance Air, the regional arm of Air India. PM Narendra Modi handed over boarding passes to some of passengers travelling via the first UDAN flight at the Shimla airport.Tomorrow PM @narendramodi will flag off the first UDAN flight under the Regional Connectivity Scheme, on Shimla-Delhi sector.Air India yesterday opened bookings for the first launch flight from Shimla to Delhi with all inclusive fares starting at Rs2,036.THE GREAT 'UDAN'The UDAN (Ude Desh ka Aam Naagrik) scheme seeks to make flying more affordable for the common people, holding a plan to connect over 45 unserved and under-served airports.Under UDAN, 50 per cent of the seats on each flight would have a cap of Rs 2,500 per seat/hour. The government has also extended subsidy in the form of viability gap funding to the operators flying on these routes.The scheme was launched to "make air travel accessible to citizens in regionally important cities," and has been described as "a first-of-its-kind scheme globally to stimulate regional connectivity through a market-based mechanism." Report have it the first flight today will not be flying at full capacity on its 70-seater ATR airplane because of payload restrictions related to the short Shimla airfield.|| Read more ||Udan scheme: Now you can fly to these 43 cities, see the full list hereUDAN scheme to fly hour-long flights capped at Rs 2,500 to smaller cities 


**Original Summary**
PM Narendra Modi on Thursday launched Ude Desh ka Aam Nagrik (UDAN) scheme for regional flight connectivity by flagging off the inaugural flight from Shimla to Delhi. Under UDAN, government will connect small towns by air with 50% plane seats' fare capped at?2,500 for a one-hour journey of 500 kilometres. UDAN will connect over 45 unserved and under-served airports.

**Generated Summary**
UDAN (Ude Desh Ka Aam Naagrik) scheme, launched to make air travel accessible in regionally important cities under the Regional Connectivity Scheme, took off from Shimla on Tuesday. The first flight will be operated by Alliance Air, which is the regional arm of India's Air India. Under the scheme, 50% seats would have?2,500 per seat/hour and 50% of the seats would have capped at this rate. It was also extended subsidy in form-based funding for operators flying these routes as well.



##### Example 3

**Original Text**
New Delhi, Apr 25 (PTI) The Income Tax department has issued a Rs 24,646 crore tax demand notice to Sahara Groups Aamby Valley Limited (AVL) after conducting a special audit of the company. The department, as part of a special investigation and audit into the account books of AVL, found that an income of over Rs 48,000 crore for a particular assessment year was allegedly not reflected in the record books of the firm and hence it raised a fresh tax demand and penalty amount on it. A Sahara Group spokesperson confirmed the development to PTI. "Yes, the Income Tax Department has raised Rs 48,085.79 crores to the income of the Aamby Valley Limited with a total demand of income tax of Rs 24,646.96 crores on the Aamby Valley Limited," the spokesperson said in a brief statement. Officials said the notice was issued by the taxman in January this year after the special audit of AVLs income for the Assessment Year 2012-13 found that the parent firm had allegedly floated a clutch of Special Purpose Vehicles whose incomes were later accounted on the account of AVL as they were merged with the former in due course of time. The AVL, in its income return filed for AY 2012-13, had reflected a loss of few crores but the special I-T audit brought up the added income, a senior official said. The Supreme Court, last week, had asked the Bombay High Courts official liquidator to sell the Rs 34,000 crore worth of properties of Aamby Valley owned by the Sahara Group and directed its chief Subrata Roy to personally appear before it on April 28.  


**Original Summary**
The Income Tax Department has issued a ?24,646 crore tax demand notice to Sahara Group's Aamby Valley Limited. The department's audit found that an income of over ?48,000 crore for the assessment year 2012-13 was not reflected in the record books of the firm. A week ago, the SC ordered Bombay HC to auction Sahara's Aamby Valley worth ?34,000 crore.

**Generated Summary**
the Income Tax department has issued a?24,646 crore tax demand notice to Sahara Groups Aamby Valley Limited (AVL) after conducting an audit of the company. The notice was issued in January this year after the special audit found that the parent firm had floated Special Purpose Vehicle income for the Assessment Year 2012-13 and later accounted on its account as they were merged with the former. "Yes...the Income Tax Department raised Rs48,085.79 crores to the income," he added earlier said at the notice.
