# fl

10/07

To run toy example, open two terminal, say T1 and T2

In T1, start server
```shell
python -m server
```

In T2, start client
```shell
python -m client1
```

To start the "training", in the web browser for server (http://127.0.0.1:5000/), navigate to 
```
http://127.0.0.1:5000/start
```


A sample log on webpage with 3 clients is shown as below:

#Server Side

##Server Homepage
[21:12:20]-[Server]: Server start.

[21:12:32]-[Server]: Init a ML model: CNN( (conv1): Sequential( (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)) (1): ReLU() (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) ) (conv2): Sequential( (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)) (1): ReLU() (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) ) (out): Linear(in_features=1568, out_features=10, bias=True) )

[21:12:32]-[Server]: Send model to http://127.0.0.1:5001/client-receive

[21:12:46]-[Server]: Receive data from client : 0

[21:12:46]-[Server]: Send model to http://127.0.0.1:5002/client-receive

[21:13:00]-[Server]: Receive data from client : 1

[21:13:00]-[Server]: Send model to http://127.0.0.1:5003/client-receive

[21:13:15]-[Server]: Receive data from client : 2

[21:13:15]-[Server]: Performing FedAvg for epoch: 0

[21:13:15]-[Server]: Remaining global epoch: 9

[21:13:15]-[Server]: Send to http://127.0.0.1:5001/client-receive with new model

[21:13:29]-[Server]: Receive data from client : 0

[21:13:29]-[Server]: Send to http://127.0.0.1:5002/client-receive with new model

[21:13:44]-[Server]: Receive data from client : 1

[21:13:44]-[Server]: Send to http://127.0.0.1:5003/client-receive with new model

[21:13:58]-[Server]: Receive data from client : 2

[21:13:58]-[Server]: Performing FedAvg for epoch: 1

[21:13:58]-[Server]: Remaining global epoch: 8

[21:13:58]-[Server]: Send to http://127.0.0.1:5001/client-receive with new model

[21:14:12]-[Server]: Receive data from client : 0

[21:14:12]-[Server]: Send to http://127.0.0.1:5002/client-receive with new model

[21:14:27]-[Server]: Receive data from client : 1

[21:14:27]-[Server]: Send to http://127.0.0.1:5003/client-receive with new model

[21:14:42]-[Server]: Receive data from client : 2

[21:14:42]-[Server]: Performing FedAvg for epoch: 2

[21:14:42]-[Server]: Federated Learning is finished with 3 epochs and accuracy is 0.98


#Client Side

##Client 1 Homepage
[21:12:24]-[Client 1]: Client start.

[21:12:32]-[Client 1]: Receive model from server

[21:12:32]-[Client 1]: Local training starts

[21:12:37]-[Client 1]: Local epoch is 1. Loss is 0.10974124073982239

[21:12:43]-[Client 1]: Local epoch is 2. Loss is 0.11847575008869171

[21:12:46]-[Client 1]: Send model from cilent 1

[21:13:15]-[Client 1]: Receive model from server

[21:13:15]-[Client 1]: Local training starts

[21:13:20]-[Client 1]: Local epoch is 1. Loss is 0.11238197237253189

[21:13:26]-[Client 1]: Local epoch is 2. Loss is 0.0225856713950634

[21:13:29]-[Client 1]: Send model from cilent 1

[21:13:58]-[Client 1]: Receive model from server

[21:13:58]-[Client 1]: Local training starts

[21:14:03]-[Client 1]: Local epoch is 1. Loss is 0.030502114444971085

[21:14:09]-[Client 1]: Local epoch is 2. Loss is 0.07069046795368195

[21:14:12]-[Client 1]: Send model from cilent 1
