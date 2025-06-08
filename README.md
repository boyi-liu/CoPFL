# Co-PFL

An easy-to-use Co-PFL research platform.


## Guidance

### Step 1: Prepare your data

You only need to tune `dataset/config.yaml` to modify the config.

Then you can run dataset-specific file to generate dataset.

```bash
python generate_mnist.py
```



### Step 2: Implement your algorithm 

#### Step 2.1 Create your file

Create a new file `{NAME}.py` inside the path `alg`.



#### Step 2.2 Extend the basic Client and Server

Extend the Client and Server class in `alg/base.py`:

```python
from alg.base import BaseServer, BaseClient

class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)

class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
```


#### Step 2.3 Config your algorithm-specific hyperparameters

And all **general** args could be found in `utils/options.py`.

For **algorithm-specific** hyperparameters, it is recommended to add a `add_args()` function inside your file.

```python
def add_args(parser):
    parser.add_argument('--{your_param}', type=int, default=1)
    return parser.parse_args()
```



#### Step 2.4 Implement your algorithms

> ‼️NOTE: We claim that each algorithm should **overwrite the function** `run()`, because it stands for the main workflow of your algorithm.

The `run()` follows a basic pipeline of:

```python
from alg.base import BaseServer, BaseClient
from utils.time_utils import time_record

class Client(BaseClient):
    @time_record
    def run(self):
        self.train()

class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
```

💡***You can overwrite or add any function as you want then!***



### Step 3: Run your code!

Now it is time to run your code!

#### Step 3.1: Config your hyperparamters

There are three places to config your hyperparameters:

+ ⭐️⭐️⭐️ Highest priority: Your bash to run the code, for example, `python main.py --total_num 10`
+ ⭐️⭐️ Medium priority: The content in `config.yaml`
+ ⭐️ Lowest priority: The default setup in `utils/options.py`

>  The priority means that, if you change `total_num` to `10` in your bash, it will **overwrite** that in `config.yaml` and `utils/options.py`.



#### Step 3.2 Run your code!

```bash
python main.py --{your args1} {your args1's value} --{your args2} {your args2's value}
```

