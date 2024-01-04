from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# URL of the Streamlit app
streamlit_url = "http://localhost:8501/"

# Set up Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode (without opening a window)
driver = webdriver.Chrome(options=chrome_options)

# Open the Streamlit app URL
driver.get(streamlit_url)

# Wait for the Streamlit app to load (adjust sleep time based on your app's load time)
time.sleep(10)

# Capture the Streamlit app content as HTML
html_content = driver.page_source

# Save the HTML content to a file
with open("streamlit_app.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# Close the browser session
driver.quit()
