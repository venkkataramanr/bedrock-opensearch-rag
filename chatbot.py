import gradio as gr
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
import boto3
import json

with open('host.txt', 'r') as fp:
    host = fp.read()
region = "us-east-1"
service = "aoss"
session = boto3.Session(profile_name="rapid-innovation-dev")
credentials = session.get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

os_client = OpenSearch(
    hosts = [{"host": host, "port": 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    pool_maxsize = 20
)
 
bedrock = session.client(
 service_name='bedrock-runtime',
 region_name='us-east-1'
)

def generate_context(text):
    body=json.dumps({"inputText": text})
    response = bedrock.invoke_model(body=body, modelId='amazon.titan-embed-text-v1', accept='*/*', contentType='application/json')
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    document = {
        "size": 15,
        "_source": {"excludes": ["nominee_vector"]},
        "query": {
            "knn": {
                 "nominee_vector": {
                     "vector": embedding,
                     "k":15
                 }
            }
        }
    }
    response = os_client.search(
    body = document,
    index = "oscars-index"
    )
    data=response['hits']['hits']
    context = ''
    for item in data:
        context += item['_source']['nominee_text'] + '\n'
    return context

def invoke_titan(prompt):
    config={
      "maxTokenCount": 1000,
      "stopSequences": [],
      "temperature":0.1,
      "topP":1
    }
    body = json.dumps({'inputText': prompt,'textGenerationConfig':config})

    response = bedrock.invoke_model( 
        modelId='amazon.titan-tg1-large', 
        body=body,
        contentType='application/json',
        accept='application/json',
        # guardrailIdentifier='arn:aws:bedrock:us-east-1:683883881884:guardrail/64blac89050a',
        # guardrailVersion='1'
    )
    response_body = json.loads(response.get('body').read())
    print(response_body)
    return response_body.get('results')[0].get('outputText')

def build_prompt(message, history):
    context=generate_context(message)
    prompt=f'Context - {context}\nBased on the above context, answer this question - {message}'
    print(prompt)
    return invoke_titan(prompt)

gr.ChatInterface(
    build_prompt,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
    title="Oscar 2023 Bot - Amazon Bedrock and Titan",
    theme="soft",
    examples=[
        "Who won the best music award?", 
        "Which award did Avatar win?",
        "Who won the Best Actor award in a supporting role?",
        "Who is the lyricist for the song Natu Natu from RRR?",
        "Which was the Best International Feature Film?",
        ],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()
