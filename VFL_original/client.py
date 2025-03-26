from phe import paillier
import numpy as np
import math


class Client:
    def __init__(self, config):
        ## 模型参数
        self.config = config
        ## 中间计算结果
        self.data = {}
        ## 与其他节点的连接状况
        self.other_client = {}
    
    ## 与其他参与方建立连接
    def connect(self, client_name, target_client):
        self.other_client[client_name] = target_client
    
    ## 向特定参与方发送数据
    def send_data(self, data, target_client):
        target_client.data.update(data)



class ClientC(Client):
    """
    辅助节点,Client C as trusted dealer.
    """
    def __init__(self, A_d_shape, B_d_shape, config):
        super().__init__(config)
        self.A_data_shape = A_d_shape
        self.B_data_shape = B_d_shape
        self.public_key = None
        self.private_key = None
        self.epoch = 0
        ## 保存训练中的损失值（泰展开近似）
        self.loss = []

    
    def task_1(self, client_A_name, client_B_name):
        """
        生成Paillier的密钥对
        """
        try:
            public_key, private_key = paillier.generate_paillier_keypair()
            self.public_key = public_key
            self.private_key = private_key
        except Exception as e:
            print("C step 1 error 1: %s" % e)

        data_to_AB = {"public_key": public_key}
        self.send_data(data_to_AB, self.other_client[client_A_name])
        self.send_data(data_to_AB, self.other_client[client_B_name])
        return
    
    def task_2(self,client_A_name,client_B_name):
        """
        解密由A,B发来的加密梯度和loss,step4
        """
        dt = self.data
        assert "encrypted_loss" in dt.keys(), "Error: 'encrypted_loss' from A in step 2 not successfully received."
        assert "encrypted_gradient_B" in dt.keys(), "Error: 'encrypted_gradient_B' from B in step 3 not successfully received."
        assert "encrypted_gradient_A" in dt.keys(), "Error: 'encrypted_gradient_A' from A in step 2 not successfully received."

        encrypted_loss = dt["encrypted_loss"]
        encrypted_gradient_B = dt["encrypted_gradient_B"]
        encrypted_gradient_A = dt["encrypted_gradient_A"]

        loss = self.private_key.decrypt(encrypted_loss)
        loss = loss/self.A_data_shape[0]+math.log(2)

        gradient_B = np.asarray([self.private_key.decrypt(i) for i in encrypted_gradient_B])
        gradient_A = np.asarray([self.private_key.decrypt(i) for i in encrypted_gradient_A])
        self.epoch += 1

        print("epoch{} loss: {}".format(self.epoch,loss))

        data_to_A = {"gradient_A":gradient_A}
        data_to_B = {"gradient_B":gradient_B}

        self.send_data(data_to_A,self.other_client[client_A_name])
        self.send_data(data_to_B,self.other_client[client_B_name])



class ClientA(Client):
    """主动方"""
    def __init__(self, X, y, config):
        super().__init__(config)
        self.X = X
        self.y = y
        self.weights = np.zeros(X.shape[1])
        
    def compute_z_a(self):
        z_a = np.dot(self.X, self.weights)
        return z_a
    
    	
    def update_weight(self, dJ_a):
        """
        参数的更新
        """
        self.weights = self.weights - self.config["lr"] * dJ_a / len(self.X)
        return
    
    
	## 加密梯度的计算，对应step4
    def compute_encrypted_dJ_a(self, encrypted_u):
        encrypted_dJ_a = self.X.T.dot(encrypted_u) + self.config['lambda'] * self.weights
        return encrypted_dJ_a
    
    
    #step2
    def task_1(self,client_B_name,client_C_name):
        """
        计算加密的loss,g_a,和用于计算梯度的[d]
        """
        try:
            dt = self.data
            assert "public_key" in dt.keys(), "Error: 'public_key' from C in step 1 not successfully received."
            pk = dt['public_key']
        except Exception as e:
            print("A step 1 exception: %s" % e)
        try:
            z_a = self.compute_z_a()
            z_a_square = z_a ** 2
            encrypted_z_a = np.asarray([pk.encrypt(x) for x in z_a])
            encrypted_z_a_square = np.asarray([pk.encrypt(x) for x in z_a_square])
            dt.update({"z_a": z_a})
        except Exception as e:
            print("Wrong 1 in A: %s" % e)

        ##计算加密loss，loss为了决定啥时候停止训练
        encrypted_z_b = dt["encrypted_z_b"]
        encrypted_z_b_square = dt["encrypted_z_b_square"]

        enctyted_z = encrypted_z_a + encrypted_z_b
        encrypted_z_square = encrypted_z_b_square + encrypted_z_a_square + 2*z_a*(encrypted_z_b)
        
        encrypted_loss = np.sum(0.125*encrypted_z_square-0.5*self.y*enctyted_z)    #为了简化,其他项在C解密之后计算

        ##计算残差项d，算梯度
        encrypted_d = 0.25 * enctyted_z - 0.5 * np.asarray([pk.encrypt(np.float64(i)) for i in self.y])

        dt.update({"encrypted_loss":encrypted_loss,"encrypted_d": encrypted_d})
        
        #计算自己的梯度
        encrypted_gradient_A = self.X.T.dot(encrypted_d) + self.config['lambda'] * self.weights

        data_to_C = {"encrypted_loss":encrypted_loss,"encrypted_gradient_A":encrypted_gradient_A}
        self.send_data(data_to_C,self.other_client[client_C_name])

        data_to_B = {"encrypted_d":encrypted_d}
        self.send_data(data_to_B,self.other_client[client_B_name])

    ##step5,更新本地参数
    def task_2(self):
        """
        A更新自己的参数
        """
        dt = self.data
        assert "gradient_A" in dt.keys(), "Error: 'gradient_A' from C in step 4 not successfully received."
        self.update_weight(dt["gradient_A"])
        print(f"A weight: {self.weights}")




class ClientB(Client):
    """参与方"""
    def __init__(self, X, config):
        super().__init__(config)
        self.X = X
        self.weights = np.zeros(X.shape[1])
        self.data = {}
        
    
    def compute_z_b(self):
        z_b = np.dot(self.X, self.weights)  
        return z_b

    def compute_encrypted_dJ_b(self, encrypted_u):
        """
        计算B的加密梯度
        """      
        encrypted_dJ_b = self.X.T.dot(encrypted_u) + self.config['lambda'] * self.weights
        return encrypted_dJ_b

    def update_weight(self, dJ_b):
        """
        更新本地参数
        """
        self.weights = self.weights - self.config["lr"] * dJ_b / len(self.X)

    #step1
    def task_1(self,client_A_name):
        """
        B生成自己的[W*X]和[(W*X)**2],发给A
        """
        dt = self.data
        assert "public_key" in dt.keys(), "Error: 'public_key' from C in step 1 not successfully received."
        pk = dt["public_key"]
        z_b = self.compute_z_b()
        z_b_square = z_b ** 2
        try:
            encrypted_z_b = np.asarray([pk.encrypt(x) for x in z_b])
            encrypted_z_b_square = np.asarray([pk.encrypt(x) for x in z_b_square])
        except Exception as e:
            print("Encypt fail, Wrong 1 in B: %s" % e)
        dt.update({"encrypted_z_b": encrypted_z_b})
        data_to_A = {"encrypted_z_b": encrypted_z_b,"encrypted_z_b_square":encrypted_z_b_square}
        self.send_data(data_to_A,self.other_client[client_A_name])

    #step3
    def task_2(self,client_C_name):
        """
        B计算自己的加密梯度
        """
        dt = self.data
        assert "encrypted_d" in dt.keys(), "Error: 'encrypted_d' from A in step3 not successfully received."

        #计算自己的梯度
        encrypted_d = dt["encrypted_d"]
        encrypted_gradient_B = self.X.T.dot(encrypted_d) + self.config['lambda'] * self.weights

        data_to_C = {"encrypted_gradient_B":encrypted_gradient_B}

        self.send_data(data_to_C,self.other_client[client_C_name])

    ##step5,更新本地梯度
    def task_3(self):
        """
        B更新自己的参数
        """
        dt = self.data
        assert "gradient_B" in dt.keys(), "Error: 'gradient_B' from C in step 4 not successfully received."
        self.update_weight(dt["gradient_B"])
        print(f"B weight: {self.weights}")