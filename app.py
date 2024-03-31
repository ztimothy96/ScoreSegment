import cv2
import pathlib
import shutil
import streamlit as st
from pdf2image import convert_from_bytes
from segment import get_staves

ARCHIVE_NAME = 'staves'
ARCHIVE_FMT = 'zip'
OUTPUT_FOLDER = 'output/'
IMS_FOLDER = 'ims/'
out_path = pathlib.Path(OUTPUT_FOLDER)
ims_path = out_path / IMS_FOLDER
if out_path.exists() and out_path.is_dir():
    shutil.rmtree(out_path) # clean up previous outputs
ims_path.mkdir(parents=True)

st.title('ScoreSegment')

st.markdown('''Welcome to ScoreSegment!
            This app takes in computer-generated sheet music and segments it into staves.
            This can be useful for creating musical score videos.''')

debug = st.toggle('Show segmenting process', key='debug')

uploaded_file = st.file_uploader('Choose a PDF file', type='pdf')

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    staves = []
    pages = convert_from_bytes(file_bytes, fmt='png')

    for page in pages:
        new_staves = get_staves(page)
        staves.extend(new_staves)
    
    for i, staff in enumerate(staves):       
        fn = 'staff{}.png'.format(i+1)
        file_path = ims_path / fn
        cv2.imwrite(str(file_path), staff)

    archive_path = out_path / ARCHIVE_NAME
    shutil.make_archive(archive_path, ARCHIVE_FMT, root_dir=ims_path)

    archive_file_name =  (ARCHIVE_NAME + '.' + ARCHIVE_FMT)
    archive_path = out_path / archive_file_name

    with open(str(archive_path), 'rb') as data:
        st.download_button(
            label='Download images here',
            data=data,
            file_name=archive_file_name
        )