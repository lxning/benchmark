import json
import os
import time

import boto3
from locust import HttpUser, task, events, between, constant
from botocore.config import Config
import numpy as np 

class SageMakerConfig:

    def __init__(self):
        self.__config__ = None

    @property
    def data_file(self):
        return self.config["dataFile"]

    @property
    def content_type(self):
        return self.config["contentType"]

    @property
    def show_endpoint_response(self):
        return self.config["showEndpointResponse"]

    @property
    def config(self):
        self.__config__ = self.__config__ or self.load_config()
        return self.__config__

    def load_config(self):
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_file, "r") as c:
            return json.loads(c.read())


class SageMakerEndpointTastSet(HttpUser):
    
    #wait_time = between (0.95, 1.05)
    wait_time = constant(0)
    
    def __init__(self, parent):
        super().__init__(parent)
        self.config = SageMakerConfig()
        
        # Endpoint
        self.region = self.client.base_url.split("://")[1].split(".")[2]
        self.endpointname = self.client.base_url.split("/")[-2]
        
        config = Config(
           retries = {
              'max_attempts': 0
           }
        )

        self.sagemaker_client = boto3.client('sagemaker-runtime', config=config)
        
        # Data
        #data_file_full_path = os.path.join(os.path.dirname(__file__), self.config.data_file)
        #with open(data_file_full_path, "rb") as f:
        #    self.payload = f.read()

    @task
    def test_invoke(self):
        #print(self.config.data_file)
        #Invoke sagemaker endpoint via the locust wrapper to track request times
        # data_ssd
        #self.payload = self.get_dali_ssd_input(1)

        # data_bert
        self.payload = self.get_bert_input(1)
        
        # data_noop
        #self.payload = self.get_noop_input(1)
        
        response = self._locust_wrapper(self._invoke_endpoint, self.payload)
        response_body = response["Body"].read()
        
        if self.config.show_endpoint_response:
            print(response_body)
            
    def _invoke_endpoint(self, payload_bytes):
        response = self.sagemaker_client.invoke_endpoint(
            EndpointName=self.endpointname,
            Body=payload_bytes,
            ContentType=self.config.content_type
        )

        return response

    def _locust_wrapper(self, func, *args, **kwargs):
        """
Locust wrapper so that the func fires the sucess and failure events for custom boto3 client
        :param func: The function to invoke
        :param args: args to use
        :param kwargs:
        :return:
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            total_time = int((time.time() - start_time) * 1000)
            events.request_success.fire(request_type="boto3", name="invoke_endpoint", response_time=total_time,
                                        response_length=0)

            return result
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            events.request_failure.fire(request_type="boto3", name="invoke_endpoint", response_time=total_time,
                                        response_length=0,
                                        exception=e)

            raise e

    def get_dali_ssd_input(self, batch_size) :
        image_data = []
        image_shape = []
        with open('ssd-01.jpg', "rb") as fd:
            image_data_ndarray = np.array(list(fd.read())).astype(np.uint8)
            image_shape = list(image_data_ndarray.shape)
            
        repeated_image_data = []
        for idx in range(batch_size):
            repeated_image_data.append(image_data_ndarray.tolist())
    
        image_shape.insert(0, batch_size)
        dali_ssd_input = [{
            "name":"IMAGE",
            "shape":image_shape,
            "datatype":"UINT8", 
            "data":repeated_image_data
        }]
        return json.dumps({"inputs": dali_ssd_input})
    
    def get_bert_input(self, batch_size) :
        seed = np.array(list("Bloomberg has decided to publish a new report on the global economy. "), dtype=object).tolist()
        seeds = []
        for i in range(batch_size):
            seeds.append(seed)

        bert_input = [{
            "name": "SENTENCE",
            "shape":[batch_size, len(seed)],
            "datatype":"BYTES",
            "data": seeds
        }]
        return json.dumps({"inputs": bert_input})

    def get_noop_input(self, batch_size):
        seed =np.random.rand(1).astype(dtype=np.float32).tolist()
        seeds = []
        for i in range(batch_size):
            seeds.extend(seed)
        noop_input = [{
                "name": "input",
                "shape": [batch_size, 1],
                "datatype": "FP32",
                "data": seeds 
        }]
        return json.dumps({"inputs": noop_input})
