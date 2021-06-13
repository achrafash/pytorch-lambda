import time
import json

from model import translate


def lambda_handler(event, context):
    body = json.loads(event['body'])
    sentence = body['sentence']
    
    start_time = time.time()
    translated_sentence = translate(sentence)
    end_time = time.time()
    
    return {
        'statusCode': 200,
        'body': json.dumps({
                'fr': sentence, 
                'en': translated_sentence,
                'duration': end_time - start_time
        })
    }