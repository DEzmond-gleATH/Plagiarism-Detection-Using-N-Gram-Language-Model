import re
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from nltk.util import pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import WittenBellInterpolated
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import webbrowser
from IPython.display import display, HTML
import os

def plagde(text_input, file_contents):
   # Function to retrieve search results from Google
    def get_search_results(query, num_results=10):
        return [url for url in search(query, num_results=num_results)]

    # Function to extract webpage title from URL
    def get_webpage_title(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            title_element = soup.find('title')
            if title_element:
                return title_element.get_text()
        except:
            pass
        return "N/A"

    train_query = text_input
    num_results = 10

    # Function to extract webpage content
    def extract_webpage_content(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.extract()
            # Extract text content
            text = soup.get_text()
            # Preprocess the text
            text = re.sub(r'[^\w\s]', '', text)
            text = text.lower()
            return text
        except:
            return "Error: Unable to extract webpage content."

    # Retrieve the top search results
    search_results = get_search_results(train_query, num_results=num_results)

    # Find and print the webpage titles
    webpage_titles = []
    for url in search_results:
        title = get_webpage_title(url)
        webpage_titles.append(title)

    # Print the webpage titles
    for i, title in enumerate(webpage_titles, start=1):
        print(f"Webpage {i} title: {title}")

    # Print the search results
    for i, result in enumerate(search_results, start=1):
        print(f"Result {i}: {result}")
    # Use the obtained results
    train_urls = result[:num_results]
    # Training data
    train_data = []
    train_urls = search_results
    for url in train_urls:
        content = extract_webpage_content(url)
        train_data.append(content)

    # Set n-gram number
    n = 4

    # Initialize language models
    models = []
    for data in train_data:
        # Tokenize and pad the text
        training_data = list(pad_sequence(word_tokenize(data), n, pad_left=True, left_pad_symbol="<s>"))
        # Generate n-grams
        ngrams = list(everygrams(training_data, max_len=n))
        # Build language models
        model = WittenBellInterpolated(n)
        model.fit([ngrams], vocabulary_text=training_data)
        print(model.vocab)
        models.append(model)

    # Testing data file
    #test_data_file = r'C:\Users\pc\Desktop\Major Project\Project_plague\student essay\student2.txt'

    # Read testing data
    #with open(test_data_file, encoding="utf8") as f:
    test_text = file_contents.lower()
    test_text = re.sub(r'[^\w\s]', '', test_text)

    # Tokenize and pad the text
    testing_data = list(pad_sequence(word_tokenize(test_text), n, pad_left=True, left_pad_symbol="<s>"))
    print("Length of test data:", len(testing_data))

    # Assign scores
    scores = []
    for model in models:
        model_scores = []
        for i, item in enumerate(testing_data[n-1:]):
            s = model.score(item, testing_data[i:i+n-1])
            model_scores.append(s)
        scores.append(model_scores)

    scores_np = np.array(scores)

    # Set width and height
    width = 8
    height = np.ceil(len(testing_data) / width).astype("int32")
    print("Width, Height:", width, ",", height)

    # Copy scores to rectangular blank array
    a = np.zeros(width * height)
    a[:len(scores_np[0])] = scores_np[0]
    diff = len(a) - len(scores_np[0])

    # Apply Gaussian smoothing for aesthetics
    a = gaussian_filter(a, sigma=1.0)

    # Reshape to fit rectangle
    a = a.reshape(-1, width)

    # Format labels
    labels = [" ".join(testing_data[i:i+width]) for i in range(n-1, len(testing_data), width)]
    labels_individual = [x.split() for x in labels]
    labels_individual[-1] += [""]*diff
    labels = [f"{x:60.60}" for x in labels]

    # Calculate total plagiarized words and corresponding URL
    max_plagiarized_words = 0
    max_plagiarized_words_url = ""

    # Generate separate heatmaps for each webpage
    heatmaps = []
    for i, model_scores in enumerate(scores):
        # Copy scores to rectangular blank array
        a = np.zeros(width * height)
        a[:len(model_scores)] = model_scores
        diff = len(a) - len(model_scores)

        # Apply Gaussian smoothing for aesthetics
        a = gaussian_filter(a, sigma=1.0)

        # Reshape to fit rectangle
        a = a.reshape(-1, width)

        # Create heatmap
        heatmap = go.Figure(data=go.Heatmap(
            z=a, x0=0, dx=1,
            y=labels, zmin=0, zmax=1,
            customdata=labels_individual,
            hovertemplate='%{customdata} <br><b>Score:%{z:.3f}<extra></extra>',
            colorscale="burg"))

        heatmap.update_layout(
        height=height * 28,
        width=1000,
        font={"family": "Courier New", "color": "white"},
        plot_bgcolor="#3A3A3A",
        paper_bgcolor="#3A3A3A"
        )
        heatmap['layout']['yaxis']['autorange'] = "reversed"
        heatmaps.append(heatmap)

    # Calculate total plagiarized words and corresponding URL
        plagiarized_words = np.sum(np.array(model_scores) > 0.5)  # Count scores above 0.5 as plagiarized
        if plagiarized_words > max_plagiarized_words:
            max_plagiarized_words = plagiarized_words
            max_plagiarized_words_url = train_urls[i]

    # Calculate percentage of plagiarized score
    total_words = len(testing_data)
    percentage_plagiarized = (max_plagiarized_words / total_words) * 100

    # Generate HTML page with heatmaps and URLs
    html_content = ""

    # Create summary section
    html_content += "<hr style='border-top: 2px solid black; margin-bottom: 20px;'>"
    html_content += "<h1>Summary</h1>"
    html_content += "<h4>Webpage with the most plagiarized words is:</h4>"
    html_content += f"<h4>URL: <a href='{max_plagiarized_words_url}' style='color: #00C5C8;'>{max_plagiarized_words_url}</a></h4>"
    html_content += f"<h4>Number of plagiarized words: {max_plagiarized_words}</h4>"
    html_content += f"<h4>Percentage of plagiarized score: {percentage_plagiarized:.2f}%</h4>"
    html_content += "<hr style='border-top: 2px solid black; margin-top: 20px;'>"



    for i, heatmap in enumerate(heatmaps):
        url = train_urls[i]
        html_content += f"<h1>Webpage {i+1}</h1>"
        html_content += f"<div><h4>Heatmap for '{webpage_titles[i]}'</h4>"
        html_content += f"<h4>URL: <a href='{url}' style='color: #00C5C8;'>{url}</a></h4>"
        # Calculate percentage of plagiarized score for the current heatmap
        total_words = len(testing_data)
        plagiarized_words = np.sum(np.array(scores[i]) > 0.5)  # Count scores above 0.5 as plagiarized
        percentage_plagiarized = (plagiarized_words / total_words) * 100
        html_content += f"<h4>Percentage of plagiarism: {percentage_plagiarized:.2f}%</h4>"
        html_content += heatmap.to_html(full_html=False, include_plotlyjs='cdn')
        html_content += "<hr style='border-top: 1px solid black; margin-bottom: 10px;'>"
        
    # Display the HTML page
    display(HTML(html_content))

    # Specify the folder path where you want to save the HTML page
    folder_path = "C:/Users/pc/Desktop/Project"

    # Create the full file path including the folder and file name
    file_path = os.path.join(folder_path, "heatmap_output.html")

    # Save the HTML page to the specified file path
    with open(file_path, 'w') as f:
        f.write(html_content)


    # Open the HTML page in a web browser
    webbrowser.open('heatmap_output.html')


    # Specify the paths of the input HTML files and the output HTML file
    file1_path = 'C:/Users/pc/Desktop/Project/templates/result3.html'
    file2_path = 'C:/Users/pc/Desktop/Project/templates/heatmap_output.html'
    output_file_path = 'C:/Users/pc/Desktop/Project/templates/heatmap_output.html'

    def combine_html_files(file1_path, file2_path, output_file_path):
        # Read the contents of the first HTML file
        with open(file1_path, 'r') as file1:
            content1 = file1.read()

        # Read the contents of the second HTML file
        with open(file2_path, 'r') as file2:
            content2 = file2.read()

        # Combine the contents of both files
        combined_content = content1 + content2 + "</div></body></html>"

        # Write the combined content to a new HTML file
        with open(output_file_path, 'w') as output_file:
            output_file.write(combined_content)

    # Call the function to combine the HTML files
    combine_html_files(file1_path, file2_path, output_file_path)
