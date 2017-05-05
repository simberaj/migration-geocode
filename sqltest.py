import sqlite3, os

STDOUT_ENCODING = 'cp852'

dbname = input('Enter database name: ')
path = os.path.join(os.getcwd(), dbname + '.db')

con = sqlite3.connect(path)
con.isolation_level = None
cur = con.cursor()

buff = ""

print("Enter your SQL commands to execute in sqlite3.")
print("Enter a blank line to exit.")

while True:
    line = input()
    if line == "":
        break
    buff += line
    if sqlite3.complete_statement(buff):
        try:
            buff = buff.strip()
            cur.execute(buff)

            if buff.lstrip().upper().startswith("SELECT"):
                print(str(cur.fetchall()).encode(STDOUT_ENCODING, errors='replace').decode(STDOUT_ENCODING))
        except sqlite3.Error as e:
            print("An error occurred:", e.args[0])
        buff = ""

con.close()
