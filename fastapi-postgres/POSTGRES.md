Interface: PGAdmin

SELECT * FROM products;

DESC - Descending order
ASC - Ascending order
LIMIT - Limit the number of rows returned

SELECT * FROM products ORDER BY price DESC, id ASC; "id ASC" means that if the price is the same, it will sort by id in ascending order.
SELECT * FROM products ORDER BY price DESC LIMIT 10;

INSERT INTO products (name, price, inventory) VALUES ('tortilla', 4, 1000);
INSERT INTO products (price, name, inventory) VALUES (4, 'car', 1000) returning *;

Inserting Entries
INSERT INTO products (price, name, inventory) VALUES (4, 'car', 1000), (50, 'laptop', 25), (60, 'Monitor', 35) returning *;

Deleting Entries
DELETE FROM products WHERE id = 1;
DELETE FROM products WHERE id = 1 AND price = 4;
DELETE FROM products WHERE id = 1 OR price = 4;

Upating Entries
UPDATE products SET name = 'tortilla' WHERE id = 1;
UPDATE products SET name = 'tortilla', price = 5 WHERE id = 1;
UPDATE products SET is_sale = true WHERE id = 22 RETURNING *;