import opt
from encoder import *


class AttentionFusionModule(nn.Module):
    def __init__(self, feature_dim, feature_num=2):
        super(AttentionFusionModule, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(feature_num * feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_num),
            nn.Softmax(dim=-1)
        )
        self.feature_num = feature_num

    def forward(self, z: list):
        combined = torch.cat(z, dim=1)
        attention_weights = self.attention_network(combined)
        weighted_z = []
        for i in range(0, self.feature_num):
            weighted_z_i = attention_weights[:, i:i + 1] * z[i]
            weighted_z.append(weighted_z_i)

        fused_z = sum(weighted_z)

        return fused_z


def q_distribution(z, u):
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - u, 2), 2))
    q = (q.T / torch.sum(q, 1)).T
    return q


def target_distribution(Q):
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P


def reconstruction(z, a_igae, adj, ae_decoder, gae_decoder):
    x_hat = ae_decoder(z)
    z_hat, z_adj_hat = gae_decoder(z, adj)
    a_hat = a_igae + z_adj_hat
    hat = [x_hat, z_hat, a_hat]
    return hat


class DAMIC(nn.Module):
    def __init__(self, ae1, ae2, gae1, gae2, n_node=None):
        super(DAMIC, self).__init__()

        self.ae1 = ae1
        self.ae2 = ae2

        self.gae1 = gae1
        self.gae2 = gae2

        self.attention_fusion = AttentionFusionModule(opt.args.n_z, feature_num=4)
        self.attention_fusion_x1 = AttentionFusionModule(opt.args.n_z, feature_num=2)
        self.attention_fusion_x2 = AttentionFusionModule(opt.args.n_z, feature_num=2)



        self.cluster_centers = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)


    def forward(self, x1, adj1, x2, adj2, pretrain=False):
        # node embedding encoded by AE
        z_ae1 = self.ae1.encoder(x1)
        z_ae2 = self.ae2.encoder(x2)

        # node embedding encoded by IGAE
        z_igae1, a_igae1 = self.gae1.encoder(x1, adj1)
        z_igae2, a_igae2 = self.gae2.encoder(x2, adj2)

        z_fused = self.attention_fusion([z_ae1, z_igae1, z_ae2, z_igae2])
        q_z_fused = q_distribution(z_fused, self.cluster_centers)
        p_z_fused = target_distribution(q_z_fused)


        z1 = self.attention_fusion_x1([z_ae1, z_igae1])
        z2 = self.attention_fusion_x2([z_ae2, z_igae2])

        q_z1 = q_distribution(z1, self.cluster_centers)
        q_z2 = q_distribution(z2, self.cluster_centers)
        #
        hat1 = reconstruction(z1, a_igae1, adj1, self.ae1.decoder, self.gae1.decoder)
        z_fused_hat1 = reconstruction(z_fused, a_igae1, adj1, self.ae1.decoder, self.gae1.decoder)
        hat2 = reconstruction(z2, a_igae2, adj2, self.ae2.decoder, self.gae2.decoder)
        z_fused_hat2 = reconstruction(z_fused, a_igae2, adj2, self.ae2.decoder, self.gae2.decoder)

        return hat1, z_fused_hat1, hat2, z_fused_hat2, z1, z2, z_fused, q_z1, q_z2, q_z_fused, p_z_fused
