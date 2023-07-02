import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_1_size)
        self.linear2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.linear3 = nn.Linear(hidden_2_size, output_size)

    def forward(self, x):
        x = func.relu(self.linear1(x))
        x = self.linear2(x)
        x = func.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)

    def save_checkpoint(self, file_name="checkpoint.pth"):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save({
            'state_dict': self.state_dict(),
        }, file_path)

    def load_checkpoint(self, file_name="checkpoint.pth"):
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.load_state_dict(checkpoint['state_dict'])
            return True
        else:
            return False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QTraining:
    def __init__(self, model, lr, gamma):
        self.device = get_device()
        self.model = model.to(self.device)
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        prediction = self.model(state)
        target = prediction.clone()
        for index in range(len(done)):
            Q_new = reward[index]
            if not done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action[index]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
