# Intelligent Expense Tracking System 💰

An intelligent expense tracking system with predictive analytics and financial insights powered by machine learning.

## 🌟 Features

- **AI Expense Forecasting**: Predict future spending patterns using machine learning.
- **Smart Budget Recommendations**: Get personalized budgeting advice tailored to your financial habits.
- **Interactive Dashboard**: Beautiful visualizations to understand your expenses and financial trends.
- **Real-time Tracking**: Instantly log and monitor your expenses.
- **Financial Insights**: Automatic spending analysis and trends to help you make better financial decisions.

## 🧠 Machine Learning Model

The expense forecasting model uses an LSTM neural network trained on historical spending data to predict future expenses. This allows the system to accurately forecast next-day spending and analyze trends over time.

## 📚 API Documentation

The FastAPI backend provides the following endpoints:

| Endpoint                       | Method | Description                          |
|---------------------------------|--------|--------------------------------------|
| `/expenses/`                    | POST   | Add a new expense                   |
| `/expenses/`                    | GET    | Retrieve all expenses               |
| `/predict/`                     | GET    | Predict the next day's expense      |
| `/forecast/`                    | GET    | Get a 7-day expense forecast        |
| `/budget/recommendations`       | GET    | Get personalized budget advice      |
| `/analytics/summary`            | GET    | Get a summary of spending analytics |
| `/health`                       | GET    | Check service health                |

## 🖥️ Frontend Components

The Streamlit dashboard includes:

- **Expense Entry Form**:
  - Date picker
  - Category dropdown
  - Amount input
- **Analytics Dashboard**:
  - Spending by category (pie chart)
  - Monthly trends (bar chart)
  - Key metrics (cards)
- **AI Features**:
  - Next-day expense prediction
  - 7-day forecast with trend analysis
  - Personalized budget recommendations
