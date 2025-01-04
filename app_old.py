import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Определение модели
class ImprovedRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedRegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 32)
        self.layer7 = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer3(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer4(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer5(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer6(x))
        x = self.layer7(x)
        return x

# Загрузка модели
model_path = r"D:\NI_ComputerVision\L36\model_full.pth"
model = torch.load(model_path)
model.eval()

# Функция для предсказания
def predict(features):
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32)
        prediction = model(inputs.unsqueeze(0))  # Добавить измерение batch
    return prediction.item()

# Интерфейс Streamlit
st.title("Прогноз цен на недвижимость")
st.write("Введите характеристики квартиры:")

# Ввод данных
балконы = st.slider("Количество балконов", 0, 5, 0)
лоджии = st.slider("Количество лоджий", 0, 5, 0)
время_в_пути = st.slider("Время в пути до метро (минуты)", 1, 60, 15)
площадь = st.number_input("Площадь квартиры (кв.м)", 10.0, 300.0, 50.0)
этаж = st.slider("Этаж", 1, 30, 5)
всего_этажей = st.slider("Всего этажей в доме", 1, 30, 10)

# Формирование признаков
features = [
    балконы,
    лоджии,
    время_в_пути,
    площадь,
    этаж / всего_этажей,
]

# Заполнение недостающих признаков
while len(features) < 1291:
    features.append(0)

features = np.array(features, dtype=np.float32)

# Кнопка предсказания
if st.button("Прогнозировать цену"):
    цена = predict(features)
    st.success(f"Прогнозируемая цена: {цена:.2f} тыс. рублей")
