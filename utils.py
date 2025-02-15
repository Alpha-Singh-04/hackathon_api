from newspaper import Article, Config
import easyocr

def extract_article_content(url):

    # Define a user-agent to mimic a browser request
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) ' \
                 'AppleWebKit/537.36 (KHTML, like Gecko) ' \
                 'Chrome/50.0.2661.102 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent


    article = Article(url, config=config)

    try:

        article.download()
        article.parse()


        return article.title, article.text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    
def split_text_into_chunks(text, max_length=512):

    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    return chunks


