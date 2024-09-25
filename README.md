# Use_Case_Valeo
Project overview:
    This is a project to show my knowledge about deploying, fine-tuning and using a Large Language Model. This project use the DistilBERT model to say if a film review is positive or negative.


Setup instructions:
    To setup the project, first run the following code in a terminal in the folder to install all the required libraries:
    pip install -r requirements.txt
    Run once the api.py script to setup and train the model and either use it by followihng the next section or kill the APi process with Ctrl+C

User guide:
    You can rerun the api.py script and it will either retrain the model or use already computed weights if available.
    Then open the api by running the following command in a terminal :uvicorn api:app --reload
    Finally, launch the request.py script in a new terminal and use it to get the predicted results of the model.

Implementation Explanation:
    1. The chosen model was DistilBERT, for low computational ressource needs, since it is a reduction of the BERT model and can therefore be trained locally quicker and with less ressources. It is suited to the task since it used the BERT model, which is suited for natural language understanding and therefore suited for the task.

    3. The dataset used is the imdb, which can be used for non-commercial purpose, is available from the datasets library and easy to use. It consists of a trianing and a test set of each 25 000 movie reviews which are either positive or negative. It is suited to the task of NLU, is not that heavy and accessible, hence why it was used.

    5. The API was made using fastapi. It was the first time I made an API and did not even really know what it meant before. I tried to do it using online sources to train myself and with the request.py script, it is easy to make inputs once the model is trained and the server opened locally. 