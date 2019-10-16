#lang racket
(require 2htdp/batch-io)
(require "decision_functions.rkt")

;input dataset
(provide toytrain)
(define toytrain "../data/toy_train.csv")

(provide titanictrain)
(define titanictrain "../data/titanic_train.csv")

(provide mushroomtrain)
(define mushroomtrain "../data/mushrooms_train.csv")

;output tree (dot file)
(provide toyout)
(define toyout "../output/toy-decision-tree.dot")

;reading input datasets
;read the csv file myfile as a list of strings
;with each line of the original file as an element of the list
;further split each line at commas
;so then we have a list of list of strings

(define (convert ls l)
  (if (null? ls) l
      (convert (cdr ls) (append l (list (rem (car ls)))))))

(define (convert1 ls)
  (map (lambda (x) (drop x 2)) ls))

(define (rem s)
  (define l (string->list s))
  (define (reml l l1 l2)
    (cond ((null? l) (append l2 (list (list->string l1))))
          ((equal? (car l) #\,) (reml  (cdr l) '() (append l2 (list (list->string l1)))))
          (else (reml  (cdr l) (append l1 (list (car l))) l2))))
  (reml l '() '()))

(provide toy-raw)
(define toy-raw (let ((toy-data (read-csv-file toytrain)))
                  (cdr toy-data)))

(provide titanic-raw)
(define titanic-raw (let ((titanic-data (read-csv-file titanictrain)))
                      (convert1 (cdr titanic-data))))

(provide mushroom-raw)
(define mushroom-raw (let ((mushroom-data (read-csv-file mushroomtrain)))
                       (cdr mushroom-data)))



;function to convert data to internal numerical format
;(features . result)
(provide format)
(define (format data)
  (cons (change (cdr data)) (string->number (car data))))
(define (change l)
  (map (lambda (x) (string->number x)) l))
;list of (features . result)
(provide toy)
(define toy (map (lambda (x) (format x)) toy-raw))

(provide titanic)
(define titanic (map (lambda (x) (format x)) titanic-raw))

(provide mushroom)
(define mushroom (map (lambda (x) (format x)) mushroom-raw))

;============================================================================================================
;============================================================================================================
;============================================================================================================

;get fraction of result fields that are 1
;used to find probability value at leaf
(provide get-leaf-prob)
(define (get-leaf-prob data)
  (define (helper l a t)
    (cond ((null? l) (/ a t))
          ((= (cdr (car l)) 1) (helper (cdr l) (+ a 1) (+ t 1)))
          (else (helper (cdr l) a (+ t 1)))))
  (helper data 0 0)
  )

;get entropy of dataset
(provide get-entropy)
(define (get-entropy data)
  (let ((a (get-leaf-prob data)))
    (if (or (= a 0) (= a 1))
        0
        (- (/ (+ (* a (log a)) (* (- 1 a) (log (- 1 a)))) (log 2)))))
  )

;find the difference in entropy achieved
;by applying a decision function f to the data
(provide entropy-diff)
(define (entropy-diff f data)
;  (define (helper1 l l1)
;    (cond ((null? l) l1)
;          (else (helper1 (cdr l) (help (cons (car l) (f (car l))) l1)))))
;  (define (help a l2)
;    (cond ((null? l2) (list (cons (list (car a)) (cons 1 (cdr a)))))
;          ((equal? (cdr a) (cdr (cdr (car l2)))) (cons (cons (append (caar l2) (list (car a))) (cons (+ 1 (cadar l2)) (cdr a))) (cdr l2)))
;          (else (cons (car l2) (help a (cdr l2))))))
  (define (helper)
    (group-by (lambda (x) (f (car x))) data))
  (define (helper2 l)
    (foldr (lambda (x y)
             (+ (*(length x) (get-entropy x)) y)) 0 l))
  (- (get-entropy data) (/ (helper2 (helper)) (length data))))
  

;choose the decision function that most reduces entropy of the data
(provide choose-f)
(define (choose-f candidates data) ; returns a decision function
  ;(if (equal? (length candidates) 1) (car candidates)
  (let ((l (map (lambda (x) (cons (entropy-diff (cdr x) data) x)) candidates)))
    (define (maxf l a)
      (cond ((null? l) a)
            ((> (caar l) (car a)) (maxf (cdr l) (car l)))
            (else (maxf (cdr l) a))))
    (cdr (maxf l (cons -inf.0 '()))))
  )

(provide DTree)
(struct DTree (desc func kids) #:transparent)

;build a decision tree (depth limited) from the candidate decision functions and data
(provide build-tree)
(define (build-tree candidates data depth)
;  (define (split-data f data l1) 
;    (define (ins l a l1)
;        (cond ((null? l1) (list (cons (list l) a)))
;              ((equal? a (cdar l1)) (cons (cons (cons l (caar l1) ) a) (cdr l1)))
;              (else (cons (car l1) (ins l a (cdr l1))))))
;     (cond ((null? data) l1)
;          (else (split-data f (cdr data) (ins (car data) ((cdr f) (caar data)) l1)))))
  (define (split-data f data)
    (map (lambda (x) (cons x ((cdr f) (caar x)))) (group-by (lambda (x) ((cdr f) (car x))) data)))
  (define (helper2 candidates data depth)
  (cond ( (null? candidates) (DTree (number->string (get-leaf-prob data)) "" '()))
    ((= depth 0)  (DTree (number->string (get-leaf-prob data)) '() '()))
        ((= (get-leaf-prob data) 0) (DTree "0" '() '()))
        ((= (get-leaf-prob data) 1)  (DTree "1" '() '()))
        (else (let* ((f (choose-f candidates data))
                     (l (remove f candidates)) 
                     (s (split-data f data))) 
                (DTree (car f) (cdr f) (map (lambda (x) (cons (cons (get-leaf-prob (car x)) (cdr x)) (helper2 l (car x) (- depth 1)))) s)))))) 
  (cons (get-leaf-prob data) (helper2 candidates data depth)))

;(define (split-data f data l1)
;    (define (ins l a l1)
;        (cond ((null? l1) (list (cons (list l) a)))
;              ((equal? a (cdar l1)) (cons (cons (cons l (caar l1)) a) (cdr l1)))
;              (else (cons (car l1) (ins l a (cdr l1)))))) 
;   (define (helper)
;     (cond ((null? data) l1)
;          (else (split-data f (cdr data) (ins (car data) ((cdr f) (caar data)) l1)))))
;    (helper))
  

;given a test data (features only), make a decision according to a decision tree
;returns probability of the test data being classified as 1
(provide make-decision)
(define (make-decision tree test)
  
    (define (helper t1 l p)
      (match t1
        [(DTree a b '() ) (string->number a)]
        [(DTree a b l1) (let* ((x (b l))
                        (f (findf (lambda (y) (equal? x (cdar y))) l1)))
                    (if (equal? f #f)
                        0
                        (helper (cdr f) l (caar f))))]))

      (helper (cdr tree) test (car tree)))
  

;============================================================================================================
;============================================================================================================
;============================================================================================================

;;annotate list with indices
(define (pair-idx lst n)
  (if (empty? lst) `() (cons (cons (car lst) n) (pair-idx (cdr lst) (+ n 1))))
  )

;generate tree edges (parent to child) and recurse to generate sub trees
(define (dot-child children prefix tabs)
  (apply string-append (map (lambda (t) (string-append tabs "r" prefix "--" "r" prefix (~a (cdr t)) "[label=\"" (~a (cdr t)) "\"];" "\n" (dot-helper (car t) (string-append prefix (~a (cdr t))) (string-append tabs "\t")))) children))
  )

;generate tree nodes and call function to generate edges
(define (dot-helper tree prefix tabs)
  (let* ([node (match tree [(DTree d f c) (cons d c)])]
         [d (car node)]
         [c (map (lambda (x) (cdr x) ) (cdr node))])
    (string-append tabs "r" prefix "[label=\"" d "\"];" "\n\n" (dot-child (pair-idx c 0) prefix tabs))
    )
  )

;output tree (dot file)
(provide display-tree)
(define (display-tree tree dtfile)
  (write-file dtfile (string-append "graph \"decision-tree\" {" "\n" (dot-helper (cdr tree) "" "\t") "}"))
  )
;;============================================================================================================
;;============================================================================================================
;;============================================================================================================