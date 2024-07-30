import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse

# 改第一阶段
torch.manual_seed(1)
class CriticBase(nn.Module):
    def __init__(self, args):
        super(CriticBase, self).__init__()
        self.args = args
        self._define_parameters()

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        pass

    def _define_parameters(self):
        #self.parameters_all_agent = nn.ModuleList()  # do not use python list []
        #for i in range(self.args.agent_count):
        parameters_dict = nn.ModuleDict()  # do not use python dict {}
        # parameters for pre-processing observations and actions
        parameters_dict["fc_obs"] = nn.Linear(self.args.observation_dim_list, self.args.hidden_dim)
        parameters_dict["fc_action"] = nn.Linear(self.args.action_dim_list, self.args.hidden_dim)

        # parameters for hidden layers
        self._define_parameters_for_hidden_layers(parameters_dict, i)

        # parameters for generating Qvalues
        parameters_dict["Qvalue"] = nn.Linear(self.args.hidden_dim, 1)
        #self.parameters_all_agent.append(parameters_dict)

    def _forward_of_hidden_layers(self, out_obs_list, out_action_list):
        pass

    def forward(self, observation_batch_list, action_batch_list):
        # pre-process
        # out_obs_list, out_action_list = [], []
        # for i in range(self.args.agent_count):
        '''
        暂时不用
        out_obs = F.relu(self.parameters_dict["fc_obs"](observation_batch_list))  
        out_action = F.relu(self.parameters_dict["fc_action"](action_batch_list))
        '''
            # out_obs_list.append(out_obs)
            # out_action_list.append(out_action)

        # key part of difference MARL methods #
        out_hidden_list = self._forward_of_hidden_layers(observation_batch_list, action_batch_list)
        # if self.args.agent_name == "NCC_AC":
        #     out_hidden_list, C_hat_list, obs_hat_list, action_hat_list = out_hidden_list
        # elif self.args.agent_name == "Contrastive":
        #     out_hidden_list, C_hat_list = out_hidden_list

        # post-process
        #Qvalue_list = []
        #for i in range(self.args.agent_count):
        Qvalue = self.parameters_dict["Qvalue"](out_hidden_list)  # linear activation for Q-value
            #Qvalue_list.append(Qvalue)

        # if self.args.agent_name == "NCC_AC":
        #     return (Qvalue_list, C_hat_list, obs_hat_list, action_hat_list)
        # elif self.args.agent_name == "Contrastive":
        #     return (Qvalue_list, C_hat_list)
        # else:
        #     return Qvalue_list
        return Qvalue
    
class CriticAttentionalMADDPG(CriticBase):
    def __init__(self, args):
        super(CriticAttentionalMADDPG, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        hidden_dim = self.args.hidden_dim
        head_dim = hidden_dim
        encoder_input_dim = hidden_dim * (self.args.agent_count + 1) # 1 is for the action of the current agent
        decoder_input_dim = hidden_dim * (self.args.agent_count - 1) #

        parameters_dict["fc_encoder_input"] = nn.Linear(encoder_input_dim, hidden_dim)
        for k in range(self.args.head_count):
            parameters_dict["fc_encoder_head" + str(k)] = nn.Linear(hidden_dim, head_dim)

        parameters_dict["fc_decoder_input"] = nn.Linear(decoder_input_dim, head_dim)

    def _global_attention(self, encoder_H, decoder_H):
        # encoder_H has a shape (source_vector_count, batch_size, hidden_dim)
        # decoder_H has a shape (batch_size, hidden_dim)
        # scores is based on "dot-product" function, it works well for the global attention #zh-cn: 基于“点积”函数的分数，对于全局注意力效果很好
        temp_scores = torch.mul(encoder_H, decoder_H)  # (source_vector_count, batch_size, hidden_dim)
        scores = torch.sum(temp_scores, dim=2)  # (source_vector_count, batch_size)
        attention_weights = F.softmax(scores.permute(1, 0), dim=1)  # (batch_size, source_vector_count)
        attention_weights = torch.unsqueeze(attention_weights, dim=2)  # (batch_size, source_vector_count, 1)
        contextual_vector = torch.matmul(encoder_H.permute(1, 2, 0), attention_weights)  # (batch_size, hidden_dim, 1)
        contextual_vector = torch.squeeze(contextual_vector)  # (batch_size, hidden_dim)
        return contextual_vector

    # in fact, K-head module and attention module are integrated into one module
    def _attention_module(self, obs_list, action_list, agent_index):
        encoder_input_list = obs_list + [action_list[agent_index]] #obs_list : batch_size  * obs_dim #action_list : batch_size  * act_dim
        decoder_input_list = action_list[:agent_index] + action_list[agent_index + 1:]

        # generating a temp hidden layer "h" (the encoder part, refer the figure in our paper)
        encoder_input = torch.cat(encoder_input_list, dim=1) #
        encoder_h = F.relu(self.parameters_all_agent[agent_index]["fc_encoder_input"](encoder_input))

        # generating action-conditional Q-value heads (i.e., the encoder part)
        encoder_head_list = []
        for k in range(self.args.head_count):
            encoder_head = F.relu(self.parameters_all_agent[agent_index]["fc_encoder_head" + str(k)](encoder_h))
            encoder_head_list.append(encoder_head)
        encoder_heads = torch.stack(encoder_head_list, dim=0)  # (head_count, batch_size, head_dim)

        # generating a temp hidden layer "H" (the decoder part, refer the figure in our paper)
        decoder_input = torch.cat(decoder_input_list, dim=1)
        decoder_H = F.relu(self.parameters_all_agent[agent_index]["fc_decoder_input"](decoder_input))

        # generating content vector (i.e., the decoder part)
        contextual_vector = self._global_attention(encoder_heads, decoder_H)  # (batch_size, head_dim)   ###！！！！！

        # contextual_vector need to be further transformed into 1-dimension Q-value
        # this will be done by the forward() function in CriticBase()

        return contextual_vector

    def _forward_of_hidden_layers(self, out_obs_list, out_action_list):
        #out_hidden_list = []
        #for i in range(self.args.agent_count):
        out = self._attention_module(out_obs_list, out_action_list, i)
            #out_hidden_list.append(out)
        return out