# import sqlite3 module
import sqlite3
# create a connection object
conn = sqlite3.connect('Beers.db')
# create a cursor object
cur = conn.cursor()
# call the cursor's execute() method to perform SQL commands
cur.execute('SELECT * From Sells;')
for row in cur:
    print(row)
# save (commit) the changes
conn.commit()
# close the connection
conn.close()