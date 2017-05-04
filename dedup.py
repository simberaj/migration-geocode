import sys

if __name__ == '__main__':
  unique = set()
  with open(sys.argv[1], encoding='utf8') as fin:
    with open(sys.argv[2], 'w', encoding='utf8') as fout:
      # prelen = 0
      for line in fin:
        unique.add(line)
      for line in sorted(unique):
        fout.write(line)
        # if len(unique) > prelen:
          # fout.write(line)
          # prelen = len(unique)