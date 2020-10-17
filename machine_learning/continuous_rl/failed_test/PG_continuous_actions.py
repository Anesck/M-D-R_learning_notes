import gym
import numpy as np

from keras import initializers, optimizers
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda
from keras import backend as K

def actor_loss(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    action = y_true
    log_proba_dense_loss =  K.log(2*np.pi*sigma**2)/2 + (action-mu)**2/(2*sigma**2)
    return log_proba_dense_loss

class PG():
    def __init__(
            self,
            action_min,
            action_max,
            states_n,
            learning_rate=0.01
            ):
        self.action_min = action_min
        self.action_max = action_max
        self.states_n = states_n
        self.lr = learning_rate
        self.__init_net()

    def __init_net(self):
        weight_init = initializers.RandomNormal(0, 0.1)
        bias_init = initializers.Constant(0.1)
        input_layer = Input(shape=(self.states_n, ))
        hidden_layer = Dense(100, input_dim=self.states_n, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init)(input_layer)
        hidden_layer_mu = Dense(1, activation='tanh', kernel_initializer=weight_init, bias_initializer=bias_init)(hidden_layer)
        hidden_layer_sigma = Dense(1, activation='softplus', kernel_initializer=weight_init, bias_initializer=bias_init)(hidden_layer)
        output_mu = Lambda(lambda x: x*2)(hidden_layer_mu)
        output_sigma = Lambda(lambda x: x+0.1)(hidden_layer_sigma)
        output_layer = Concatenate()([output_mu, output_sigma])
        self.actor = Model(inputs=input_layer, outputs=output_layer)
        self.actor.compile(optimizer=optimizers.Adam(self.lr), loss=actor_loss)

    def choose_action(self, state):
        mu, sigma = self.actor.predict_on_batch(state.reshape(1, -1))[0]
        action = np.random.normal(mu, sigma)
        return np.clip(action, self.action_min, self.action_max), sigma

    def learn(self, state, action, reward):
        state = state.reshape(-1, self.states_n)
        reward = np.array([reward])
        loss =  self.actor.train_on_batch(state, action, sample_weight=reward)
        return loss

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    RL = PG(env.action_space.low, env.action_space.high, env.observation_space.shape[0], 0.01)
    np.set_printoptions(precision=5, suppress=True)

    step = 0
    Reward = 0
    state = env.reset()
    while True:
        env.render()
        action, sigma = RL.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        loss = RL.learn(state, action, reward/10)
        state = next_state

        Reward += reward
        step += 1
        average_reward = Reward / step
        print("\nstep: {:>7}  , reward: {:>10.5f}  , average reward: {:>10.5f}  ,loss: {:>10.5f}  , sigma: {:>.5f}".format(step, reward, average_reward, loss, sigma))

        if (step % 25 == 0 and average_reward < -6.5 and reward < -5) or \
                (step % 50 == 0 and average_reward < -6 and reward < -4) or \
                (step % 100 == 0 and average_reward < -5.5 and reward < -3) or\
                (step % 200 == 0 and reward < -4):
            state = env.reset()
    env.close()



