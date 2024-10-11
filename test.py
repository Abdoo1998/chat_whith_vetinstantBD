import mysql.connector

try:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Vetinstant@9588#!",
        database="vetinstant"
    )
    
    if connection.is_connected():
        db_info = connection.get_server_info()
        print(f"Successfully connected to MySQL database. Server version: {db_info}")
        
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE();")
        db_name = cursor.fetchone()[0]
        print(f"You're connected to database: {db_name}")
        
except mysql.connector.Error as e:
    print(f"Error connecting to MySQL database: {e}")

finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
