import os
from flask import Flask, render_template, request
from embed import Embedder

current_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,template_folder=current_dir)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     return render_template('chatbot.html')

file_name = None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global file_name
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = 'documents/'
            # Delete all files in the file path
            for filename in os.listdir(file_path):
                file_path_to_delete = os.path.join(file_path, filename)
                if os.path.isfile(file_path_to_delete):
                    os.remove(file_path_to_delete)
            file.save('documents/' + file.filename)
            embedder = Embedder(file_name=file.filename)
            file_name = file.filename
            embedder.create_vector_store()
            return render_template('chatbot.html')
    return render_template('upload.html')

@app.route('/get-response', methods=['GET'])
def get_response():
    global file_name
    query = request.args.get('query')
    print('file_name',file_name)
    embedder = Embedder(file_name=file_name)
    return embedder.chat(query)


if __name__ == '__main__':
    app.run(debug=True)