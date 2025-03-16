# Zinger Assistant Chatbot

A Streamlit-based chatbot for interior design course lead qualification.

## Deployment Steps

1. Fork this repository
2. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your forked repository
4. Add your OpenAI API key in Streamlit Cloud:
   - Go to App Settings > Secrets
   - Add `OPENAI_API_KEY = "your-api-key"`
5. Deploy!

## Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Create `.env` file with `OPENAI_API_KEY=your-api-key`
3. Run: `streamlit run ui.py`
