import torch
from model import *
from utils import *
from create_data import *


def train(data_t1, data_t2, lambda_cycle=10):
    generator_t1_to_t2.train()
    generator_t2_to_t1.train()

    discriminator_t1.train()
    discriminator_t2.train()


    optimizer_G.zero_grad()
    # Get fakes
    fake_t2 = generator_t1_to_t2(data_t1)
    fake_t2 = convert_generated_to_graph(fake_t2)  # Ensure fake_t2 is a Data object
    loss_GAN_t2 = criterion_GAN(fake_t2.edge_attr, data_t2.edge_attr).float() + frobenius_distance_loss(fake_t2.edge_attr, data_t2.edge_attr).float()

    fake_t1 = generator_t2_to_t1(data_t2)
    fake_t1 = convert_generated_to_graph(fake_t1)  # Ensure fake_t2 is a Data object
    loss_GAN_t1 = criterion_GAN(fake_t1.edge_attr, data_t1.edge_attr).float() + frobenius_distance_loss(fake_t1.edge_attr, data_t1.edge_attr).float()

    reconstructed_t1 = generator_t1_to_t2(fake_t2)
    reconstructed_t1 = convert_generated_to_graph(reconstructed_t1)
    loss_reconstructed_t1 = criterion_GAN(reconstructed_t1.edge_attr, data_t1.edge_attr).float() + frobenius_distance_loss(reconstructed_t1.edge_attr, data_t1.edge_attr).float()

    reconstructed_t2 = generator_t1_to_t2(fake_t1)
    reconstructed_t2 = convert_generated_to_graph(reconstructed_t2)
    loss_recontructed_t2 = criterion_GAN(reconstructed_t2.edge_attr, data_t2.edge_attr).float() + frobenius_distance_loss(reconstructed_t2.edge_attr, data_t2.edge_attr).float()

    total_loss = (loss_GAN_t1 + loss_GAN_t2) * 5.0 + (loss_reconstructed_t1 + loss_recontructed_t2) * lambda_cycle
    total_loss.backward()
    optimizer_G.step()

    #---------------Step 4--------------------------------
    # Training Discriminators

    # T2
    pred_real = discriminator_t2(data_t2)
    loss_t2_real = adversarial_loss(pred_real, torch.ones_like(pred_real))
    
    pred_fake = discriminator_t2(fake_t2.detach())
    loss_t2_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))


    loss_D_B = (loss_t2_real + loss_t2_fake)*0.5
    # loss_D_B = loss_t2_fake *0.5
    loss_D_B.backward()

    optimizer_D_t2.step()

    # T1
    pred_real = discriminator_t1(data_t1)
    loss_t1_real = adversarial_loss(pred_real, torch.ones_like(pred_real))

    pred_fake = discriminator_t1(fake_t1.detach())
    loss_t1_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))


    loss_D_A = (loss_t1_real + loss_t1_fake)*0.5
    loss_D_A.backward()
    optimizer_D_t1.step()

    return {"loss_G": loss_GAN_t2,
            "loss_D_B": loss_D_B,
            'loss_D_A':loss_D_A}

import itertools


def test_model(data_t1, data_t2):
    generator_t1_to_t2.eval()
    generator_t2_to_t1.eval()

    fake_Y = generator_t1_to_t2(data_t1)  # from t1 -> t2
    fake_Y = convert_generated_to_graph(fake_Y)  # from t1 -> t2

    loss_Y = criterion_GAN(fake_Y.edge_attr, data_t2.edge_attr)

    fake_X = generator_t2_to_t1(data_t2)  # from t1 -> t2
    fake_X = convert_generated_to_graph(fake_X)  # from t1 -> t2

    loss_X = criterion_GAN(fake_X.edge_attr, data_t2.edge_attr)
    return loss_Y, loss_X


generator_t1_to_t2 = Generator()
generator_t2_to_t1 = Generator()

discriminator_t2 = Discriminator()
discriminator_t1 = Discriminator()


optimizer_G = torch.optim.Adam(itertools.chain(generator_t1_to_t2.parameters(),
                                               generator_t2_to_t1.parameters()),
                               lr=0.001, betas=(0.5, 0.999))

optimizer_D_t2 = torch.optim.Adam(discriminator_t2.parameters(), lr=0.0001, weight_decay=1e-4)
optimizer_D_t1 = torch.optim.Adam(discriminator_t1.parameters(), lr=0.0001, weight_decay=1e-4)

criterion_GAN = torch.nn.L1Loss()
adversarial_loss = torch.nn.BCELoss()

train_loader, test_loader = return_dataset_all(batch_size=1)
vals = []

loss_A_to_B_list = []
loss_B_to_A_list = []
loss_D_A_list = []
loss_D_B_list = []
loss_G_list = []

total_epochs = 70
for epochs in range(1, total_epochs+1):
    loss_A_to_B, loss_B_to_A, loss_D_A, loss_D_B, loss_G = 0, 0, 0, 0, 0

    for data_t1, data_t2 in train_loader:
        data = train(data_t1, data_t2)
        loss_G += data['loss_G'].item()
        loss_D_B += data['loss_D_B'].item()
        loss_D_A += data['loss_D_A'].item()

    loss_G_list.append(loss_G/len(train_loader))
    loss_D_B_list.append(loss_D_B/len(train_loader))
    loss_D_A_list.append(loss_D_A/len(train_loader))

    print(f"Epoch:{epochs}, "
      f"Loss G:{loss_G / len(train_loader):.5f}, "
      f"Loss Disc B:{loss_D_B / len(train_loader):.5f}, "
      f"Loss Disc A:{loss_D_A / len(train_loader):.5f}")


    if epochs % total_epochs == 0 and epochs !=0:
        avg_fake_X, avg_fake_Y = [], []
        print("------------------------------")
        idx = 0
        for data_t1, data_t2 in test_loader:
            loss_Y, loss_X = test_model(data_t1, data_t2)
            avg_fake_X.append(loss_X)
            avg_fake_Y.append(loss_Y)
            idx += 1
        print("Test Results")
        print(f"Avg fake Y:{sum(avg_fake_Y)/len(avg_fake_Y)}")
        print(f"Avg fake X:{sum(avg_fake_X)/len(avg_fake_X)}")
        print("------------------------------")