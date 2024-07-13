"""
Author: Oudai Almustafa
Date: 06/26/2024
ISTA 331 Hw1

This module implements a simple recommender system using item-based collaborative filtering.
The functions interact with a bookstore database to create and manage recommendation matrices.
"""

import pandas as pd
import sqlite3
import random

def get_purchase_matrix(conn):
    query = """
    SELECT cust_id, isbn
    FROM orderitems
    INNER JOIN orders ON orderitems.order_num = orders.order_num
    ORDER BY isbn;
    """
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    
    purchase_matrix = {}
    for row in rows:
        cust_id, isbn = row
        if cust_id not in purchase_matrix:
            purchase_matrix[cust_id] = []
        if isbn not in purchase_matrix[cust_id]:
            purchase_matrix[cust_id].append(isbn)
    
    for cust_id in purchase_matrix:
        purchase_matrix[cust_id].sort()

    return purchase_matrix

def get_empty_count_matrix(conn):
    query = "SELECT DISTINCT isbn FROM orderitems ORDER BY isbn;"
    cur = conn.cursor()
    cur.execute(query)
    isbns = [row[0] for row in cur.fetchall()]
    
    count_matrix = pd.DataFrame(0, index=isbns, columns=isbns)
    return count_matrix

def fill_count_matrix(count_matrix, purchase_matrix):
    for books in purchase_matrix.values():
        for i in range(len(books)):
            count_matrix.loc[books[i], books[i]] += 1
            for j in range(i + 1, len(books)):
                count_matrix.loc[books[i], books[j]] += 1
                count_matrix.loc[books[j], books[i]] += 1
    return None

def make_probability_matrix(count_matrix):
    prob_matrix = count_matrix.copy().astype(float)
    for row in prob_matrix.index:
        total_purchases = prob_matrix.loc[row, row]
        if total_purchases > 0:
            prob_matrix.loc[row] = prob_matrix.loc[row] / total_purchases
    for row in prob_matrix.index:
        prob_matrix.loc[row, row] = -1
    return prob_matrix

def sparse_p_matrix(prob_matrix):
    sparse_matrix = {}
    for isbn in prob_matrix.index:
        sorted_probs = prob_matrix.loc[isbn].sort_values(ascending=False, kind='mergesort')
        most_likely = sorted_probs.head(15).index.tolist()
        sparse_matrix[isbn] = most_likely
    return sparse_matrix

def get_cust_id(conn):
    query = "SELECT cust_id, last, first FROM customers;"
    cur = conn.cursor()
    cur.execute(query)
    customer_data = cur.fetchall()
    
    print("CID       Name")
    print("-----     -----")
    for cust_id, last, first in customer_data:
        print(f"{cust_id:5}     {last}, {first}")
    print("---------------")
    
    try:
        cust_id = int(input("Enter customer ID: "))
    except ValueError:
        print("Invalid input. Please enter a valid customer ID.")
        return None

    if any(cust_id == row[0] for row in customer_data):
        return cust_id
    else:
        print("Customer ID not found.")
        return None

def purchase_history(cust_id, isbns, conn):
    """
    Retrieves and formats the purchase history for a given customer.

    Parameters:
    cust_id (int): The customer ID.
    isbns (list): List of ISBNs purchased by the customer.
    conn (sqlite3.Connection): SQLite connection object.

    Returns:
    str: Formatted purchase history.
    """
    query = "SELECT book_title FROM books WHERE isbn = ?;"
    cur = conn.cursor()
    
    history = []
    for isbn in isbns:
        cur.execute(query, (isbn,))
        result = cur.fetchone()
        if result:
            title = result[0]
            history.append(title)
    
    query_name = "SELECT first, last FROM customers WHERE cust_id = ?;"
    cur.execute(query_name, (cust_id,))
    customer = cur.fetchone()
    customer_name = f"{customer['first']} {customer['last']}"

    # Calculate the length of the name-based separator
    name_len = len(f"Purchase history for {customer_name}")
    separator = '-' * name_len

    history_str = f"Purchase history for {customer_name}\n"
    history_str += separator + "\n"
    history_str += "\n".join(history) + "\n"
    history_str += '-' * 40 + "\n"

    return history_str

def get_recent(cust_id, conn):
    query = """
    SELECT isbn
    FROM orderitems
    WHERE order_num = (
        SELECT order_num
        FROM orders
        WHERE cust_id = ?
        ORDER BY order_date DESC
        LIMIT 1
    );
    """
    cur = conn.cursor()
    cur.execute(query, (cust_id,))
    isbns = [row[0] for row in cur.fetchall()]
    
    if isbns:
        return isbns[random.randrange(len(isbns))]
    else:
        return None

def get_recommendation(cust_id, spm, purchased_isbns, conn):
    recent_isbn = get_recent(cust_id, conn)
    if not recent_isbn:
        return "No recent purchases found."

    query = "SELECT last, first FROM customers WHERE cust_id = ?;"
    cur = conn.cursor()
    cur.execute(query, (cust_id,))
    customer = cur.fetchone()
    customer_name = f"{customer['first']} {customer['last']}"
    
    similar_books = [isbn for isbn in spm[recent_isbn] if isbn not in purchased_isbns]
    
    if not similar_books:
        return "Out of ideas, go to Amazon\n"
    
    recommendations = similar_books[:2]
    titles = []
    for isbn in recommendations:
        cur.execute("SELECT book_title FROM books WHERE isbn = ?", (isbn,))
        title = cur.fetchone()[0]
        titles.append(title[:80])

    recommendation_str = f"Recommendations for {customer_name}\n"
    recommendation_str += "-" * (len(recommendation_str) - 1) + "\n"
    recommendation_str += "\n".join(titles) + "\n"

    return recommendation_str

def isbn_to_title(conn):
    c = conn.cursor()
    query = 'SELECT isbn, book_title FROM Books;'
    return {row['isbn']: row['book_title'] for row in c.execute(query).fetchall()}

def select_book(itt):
    isbns = sorted(itt)
    print('All books:')
    print('----------')
    for i, isbn in enumerate(isbns):
        print(' ', i, '-->', isbn, itt[isbn][:60])
    print('-' * 40)
    selection = input('Enter book number or return to quit: ')
    return isbns[int(selection)] if selection else None

def similar_books(key, cm, pm, itt, spm):
    bk_lst = []
    for isbn in cm.columns:
        if key != isbn:
            bk_lst.append((cm.loc[key, isbn], isbn))
    bk_lst.sort(reverse=True)
    print('Books similar to', itt[key] + ':')
    print('-----------------' + '-' * (len(itt[key]) + 1))
    for i in range(5):
        print(str(i) + ':')
        print(' ', bk_lst[i][0], '--', itt[bk_lst[i][1]][:80])
        print('  spm:', itt[spm[key][i]][:80])
        print('  p_matrix:', pm.loc[key, bk_lst[i][1]])

def main1():
    conn = sqlite3.connect('bookstore.db')
    conn.row_factory = sqlite3.Row
    purchase_matrix = get_purchase_matrix(conn)
    count_matrix = get_empty_count_matrix(conn)
    fill_count_matrix(count_matrix, purchase_matrix)
    p_matrix = make_probability_matrix(count_matrix)
    spm = sparse_p_matrix(p_matrix)
    
    itt = isbn_to_title(conn)
    selection = select_book(itt)
    while selection:
        similar_books(selection, count_matrix, p_matrix, itt, spm)
        input('Enter to continue:')
        selection = select_book(itt)
    
    cid = get_cust_id(conn)
    while cid:
        print()
        titles = purchase_history(cid, purchase_matrix[cid], conn)
        print(titles)
        print(get_recommendation(cid, spm, purchase_matrix[cid], conn))
        input('Enter to continue:')
        cid = get_cust_id(conn)

def main2():
    conn = sqlite3.connect('bookstore.db')
    conn.row_factory = sqlite3.Row
    
    purchase_matrix = get_purchase_matrix(conn)
    print('*' * 20, 'Purchase Matrix', '*' * 20)
    print(purchase_matrix)
    print()
    
    count_matrix = get_empty_count_matrix(conn)
    print('*' * 20, 'Empty Count Matrix', '*' * 20)
    print(count_matrix)
    print()
    
    fill_count_matrix(count_matrix, purchase_matrix)
    print('*' * 20, 'Full Count Matrix', '*' * 20)
    print(count_matrix)
    print()
    
    p_matrix = make_probability_matrix(count_matrix)
    print('*' * 20, 'Probability Matrix', '*' * 20)
    print(p_matrix)
    print()
    
    spm = sparse_p_matrix(p_matrix)
    print('*' * 20, 'Sparse Probability Matrix', '*' * 20)
    print(spm)
    print()
    
    itt = isbn_to_title(conn)
    print('*' * 20, 'itt dict', '*' * 20)
    print(itt)
    print()

if __name__ == "__main__":
    main1()
