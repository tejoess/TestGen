from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    data = {}
    if request.method == 'POST':
        # Syllabus
        if request.form.get('syllabus_enable'):
            syllabus_type = request.form.get('syllabus_type')
            if syllabus_type == 'pdf':
                file = request.files.get('syllabus_pdf')
                if file:
                    filename = file.filename
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    data['syllabus_file'] = filepath
            else:
                data['syllabus_text'] = request.form.get('syllabus_text')

        # Notes
        if request.form.get('notes_enable'):
            notes_type = request.form.get('notes_type')
            if notes_type == 'pdf':
                file = request.files.get('notes_pdf')
                if file:
                    filename = file.filename
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    data['notes_file'] = filepath
            else:
                data['notes_text'] = request.form.get('notes_text')

        # PYQs
        if request.form.get('pyqs_enable'):
            pyqs_type = request.form.get('pyqs_type')
            if pyqs_type == 'pdf':
                file = request.files.get('pyqs_pdf')
                if file:
                    filename = file.filename
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    data['pyqs_file'] = filepath
            else:
                data['pyqs_text'] = request.form.get('pyqs_text')

        # Weightage Tags
        data['weightage_tags'] = request.form.getlist('weightage_tags')

        # Sections
        sections = []
        sec_count = int(request.form.get('sections_count', 0))
        for i in range(1, sec_count + 1):
            sections.append({
                'type': request.form.get(f'section_{i}_type'),
                'count': request.form.get(f'section_{i}_count'),
                'marks': request.form.get(f'section_{i}_marks')
            })
        data['sections'] = sections

        # Save summary or pass to another page if needed
        print(data)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)