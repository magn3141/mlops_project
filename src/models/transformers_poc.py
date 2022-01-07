import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data import TextDataset

tokenizer = AutoTokenizer.from_pretrained("flax-community/dansk-gpt-wiki", model_max_length=512)
model = AutoModelForCausalLM.from_pretrained("flax-community/dansk-gpt-wiki")

dataset = TextDataset(path="./data/raw/press_conferences/", tokenizer=tokenizer, max_length=512)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train().to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-5)
for epoch in range(10):
    print("--- Epoch:",epoch,"---")
    running_loss = 0
    for i,batch in enumerate(dataloader):
        outputs = model(batch, labels=batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    avg_loss = running_loss/len(dataloader)
    print(f"Epoch training loss: {loss}")