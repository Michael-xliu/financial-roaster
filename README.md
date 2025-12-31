# 🔥 Financial Roaster

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

**Get brutally roasted for your terrible financial decisions! 💸**

Financial Roaster is an AI-powered web application that analyzes your spending habits and delivers savage, hilarious roasts about your financial choices. Upload bank statements, input spending manually, or try sample personalities to get roasted by our AI financial critic.

## ✨ Features

- **🎲 Random Sample Personalities**: Get roasted as a Tech Bro, College Student, Impulse Shopper, or Coffee Addict
- **📄 File Upload**: Upload CSV, PDF, JPEG, or PNG bank statements for analysis
- **⌨️ Manual Input**: Type your spending disasters directly for instant roasting
- **🏆 Financial Chaos Score**: Get rated 0-100 on how chaotic your spending is
- **🎨 Ultra-Premium Flip Cards**: Share your roasts with shiny 3D animated cards
- **📱 Social Media Ready**: Download cards optimized for Instagram, Twitter, and TikTok
- **⚡ Real-time Typewriter Effect**: Watch your roast appear character by character

## 🚀 Live Demo

Try it live at: **[Your Render URL will go here]**

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **AI Engine**: AWS Bedrock Claude 3 Haiku
- **Frontend**: Pure HTML/CSS/JavaScript
- **Document Processing**: PyMuPDF, EasyOCR, Tesseract
- **Deployment**: Render

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.11+
- AWS Account with Bedrock access
- Claude 3 Haiku model access in AWS Bedrock

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
AWS_REGION=us-east-1                    # Your AWS region
AWS_ACCESS_KEY_ID=your_access_key       # AWS credentials
AWS_SECRET_ACCESS_KEY=your_secret_key   # AWS credentials
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0  # Optional, uses default
BEDROCK_MAX_TOKENS=700                  # Optional, uses default
```

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/financial-roaster.git
   cd financial-roaster
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables** (see above)

4. **Run the development server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Open your browser**: http://localhost:8000

## 🌐 Deploy to Render

### One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Manual Deploy

1. **Fork this repository**

2. **Create a new Web Service on Render**:
   - Connect your GitHub account
   - Select this repository
   - Configure the service:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
     - **Environment**: Python 3.11

3. **Set Environment Variables** in Render dashboard:
   - `AWS_REGION`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `BEDROCK_MODEL_ID` (optional)
   - `BEDROCK_MAX_TOKENS` (optional)

4. **Deploy!** - Your app will be live in minutes

## 🔧 Configuration

### AWS Bedrock Setup

1. **Enable Claude 3 Haiku** in AWS Bedrock console
2. **Create IAM user** with Bedrock permissions:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "bedrock:InvokeModel"
         ],
         "Resource": "*"
       }
     ]
   }
   ```

### Supported File Formats

- **CSV**: Bank transaction exports
- **PDF**: Bank statements
- **JPEG/PNG**: Screenshots of statements

## 🎭 Sample Personalities

The app includes 4 hilarious sample personalities:

- **💻 Silicon Valley Tech Bro**: WeWork memberships, AirPods Max, productivity tools
- **🎓 Broke College Student**: Instant ramen, unused textbooks, streaming subscriptions
- **🛍️ Impulse Shopping Queen**: TikTok ads, Target hauls, "limited time offers"
- **☕ Coffee Shop Regular**: Multiple daily Starbucks visits, expensive coffee habits

## 📊 API Endpoints

- `GET /` - Main web interface
- `GET /health` - Health check
- `GET /samples` - Get sample personalities
- `POST /analyze` - Analyze transactions
- `POST /analyze-sample/{sample_id}` - Analyze sample personality
- `POST /analyze-manual` - Analyze manually entered text
- `POST /upload` - Upload and analyze file

## 🎨 Customization

### Adding New Sample Personalities

Edit `app/main.py` in the `get_sample_transactions()` function:

```python
elif sample_type == "your_personality":
    return [
        Transaction(id="s1", date="2024-01-01", amount=50.0,
                   merchant="Your Store", memo="Your spending",
                   category="Your Category", source="sample"),
        # Add more transactions...
    ]
```

### Modifying Roast Style

The AI prompts are in `core/simple_graph.py`. Adjust the system prompts to change tone:

```python
"You are a savage financial roaster. Be brutal but concise..."
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This app is for entertainment purposes only. Financial roasts are AI-generated humor and should not be considered actual financial advice. Please consult with qualified financial professionals for real financial guidance.

## 🔥 About

Built with ❤️ and a lot of ☕ by developers who've made questionable financial decisions themselves.

**Ready to get roasted? Your financial disasters await judgment! 💸🔥**

---

### 🚀 Ready to Deploy?

1. **Fork this repo**
2. **Click the Deploy to Render button**
3. **Set your AWS credentials**
4. **Watch your financial roasting app go live!**

Questions? Issues? Want to add more savage roasts? Open an issue or submit a PR!