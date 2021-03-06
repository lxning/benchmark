#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import sys
import gevent.ssl
import torch
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

def test_infer(model_name, input0_data, headers=None):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('input__0', [1, 3, 300, 300], "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)

    outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('output__1', binary_data=False))
    results = triton_client.infer(model_name,
                                  inputs,
                                  outputs=outputs,
                                  headers=headers)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable encrypted link to the server using HTTPS')
    parser.add_argument(
        '-H',
        dest='http_headers',
        metavar="HTTP_HEADER",
        required=False,
        action='append',
        help='HTTP headers to add to inference server requests. ' +
        'Format is -H"Header:Value".')

    FLAGS = parser.parse_args()
    try:
        if FLAGS.ssl:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url,
                verbose=FLAGS.verbose,
                ssl=True,
                ssl_context_factory=gevent.ssl._create_unverified_context,
                insecure=True)
        else:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "ssd_300"

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    #input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    #input0_data = np.expand_dims(input0_data, axis=0)
    #input1_data = np.full(shape=(1, 16), fill_value=-1, dtype=np.int32)

    if FLAGS.http_headers is not None:
        headers_dict = {
            l.split(':')[0]: l.split(':')[1] for l in FLAGS.http_headers
        }
    else:
        headers_dict = None
    
    #input0 = [torch.randn((1,3,300,300))]
    input0 = np.random.rand(1,3, 300, 300).astype(np.float32)
    # Infer with requested Outputs
    results = test_infer(model_name, input0, headers_dict)
    print(results.get_response())

    statistics = triton_client.get_inference_statistics(model_name=model_name,
                                                        headers=headers_dict)
    print(statistics)
    if len(statistics['model_stats']) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
