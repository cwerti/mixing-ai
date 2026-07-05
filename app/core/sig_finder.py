def find_sig():
    data = open('FlProject/template.fst', 'rb').read()
    sig = b'\xab\x2a\x00\x00\x1c\x47\x00\x00\x8e\x63\x00\x00\x00\x80\x00\x00\x72\x9c\x00\x00\xe4\xb8\x00\x00\x55\xd5\x00\x00'
    offsets = []
    idx = data.find(sig)
    while idx != -1:
        offsets.append(idx)
        idx = data.find(sig, idx + 1)
    print('Found at:', offsets)

if __name__ == '__main__':
    find_sig()
