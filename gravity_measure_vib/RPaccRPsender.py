# sender.py
def write_char(file_path, char):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(char)

path = 'C:/Users/MakarovAO/Desktop/Adamov_Kirill/gravity_measurements/gravity measure vib/testdata'
write_char("%s/stat.txt" % path, '1')