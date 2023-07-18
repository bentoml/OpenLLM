(ns openllm.api.indexed-db.core
  "This namespace is a wrapper for the IndexedDB API. It provides
   functions to create object stores, add objects to them and retrieve
   objects from them.

   The functions in this namespace are meant to be used by other
   namespaces, which will provide a higher level API for the application
   to use.

   If you stumble upon a parameter named `obj-store-fqn`, this is the fully
   qualified name of the object store. This name (or identifier rather)
   must consist of a map with two keys: `:db-name` and `:os-name`. `:db-name`
   must be a string, which identifies the database. `:os-name` must be a
   string, which is the name of the object store."
  (:require [openllm.api.log4cljs.core :refer [log]]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;             Private API            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def ^:private ^:const READ_WRITE "readwrite")
(def ^:private ^:const READ_ONLY "readonly")

(declare create-object-store!)

(def ^:private ^:const name->db
  "This map will hold the database objects for each database name.
   The database name is the key and the database object is the value.
   This map is used to prevent the creation of multiple database objects
   for the same database name. This is important, because the database
   object is the only way to interact with the database.
   
   The lookup into this map is automatically done. While using an atom
   adds state to the namespace, the performance and convinience gains
   are worth it."
  (atom {}))

(defn- idb-error-callback
  "This function is called when an error occurs during an IndexedDB
   request.
   It will log the error to the browser's console."
  [e]
  (log :error "Error during IndexedDB request" e))

(defn- create-transaction
  "Create a transaction for the object store identified by the
   `obj-store-fqn` (see namespace docstring for more information). This
   function is meant to be used by the object store ('os-*') functions.
   This function will create a transaction and return the object store,
   which can be used to interact with the database right away.

   We consider this function semi-pure since there are no *notable*
   direct side effects."
  [obj-store-fqn mode]
  (let [{:keys [db-name os-name]} obj-store-fqn
        db (get @name->db db-name)
        transaction (.transaction db #js [os-name] mode)]
    (set! (.-onerror transaction) idb-error-callback)
    (-> transaction
        (.objectStore os-name))))

(defn- on-upgrade-needed!
  "This function is called as a callback when the database is upgraded.
   It will create the object stores for the application and and save the
   backing database in the app-db for later use.

   There are two possible reasons that the database got upgraded:
   1. The database did not exist before and was created.
   2. The database existed before, but the version (and presumably the
      schema) was lower/older than the current version.

   Returns `nil`."
  [table-info user-callback e]
  (let [db (.. e -target -result)
        old-version (.-oldVersion db)
        new-version (.-version db)]
    (when (and (> 0 old-version) (nil? user-callback))
      (throw
       (ex-info "The database version was upgraded, but no 'on-upgrade-db-version' callback was provided to 'initialize!'."
                {:old-version old-version :new-version new-version})))
    (if (some? user-callback)
      (do (log :info (str "Received upgrade needed event, current version is " new-version ", old version is " old-version ". Calling user callback."))
          (user-callback old-version new-version))
      (do (log :info (str "Database and object store created. Current database version is " (.-version db) "."))
          (create-object-store! {:db-name db :os-name (:name table-info)} table-info)))
    nil))

(defn- on-initialize-success!
  "This function is called as a callback when the database is initialized.
   It will save the database object in our `name->db` atom for later use.

   Returns `nil`."
  [db-name user-callback e]
  (let [db (.. e -target -result)]
    (swap! name->db assoc db-name db)
    (user-callback)
    (log :debug "Database initialized and callback function triggered." e))
  nil)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;             Public API             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn create-object-store!
  "Create an object store identified by the `obj-store-fqn` (see namespace
   docstring for more information) inside a database. The `:os-name` key of
   the `obj-store-fqn` parameter can be chosen freely, but it must be unique
   within the database (see namespace docstring for more info) and it should
   match the `:name` key of the `table-info` parameter. The `table-info`
   parameter must be a map with the following structure:
    ```clojure
    {:name \"my-obj-store-name\"
     :index [{:name \"my-field\"
              :unique false}]}
    ```
   The `:index` key is a vector of maps, each of which will describe one
   index (aka field) of the object store. The `:name` key of the index map
   must be unique within the object store.

   Will return the object store object with an open transaction attached,
   so that it can be used right away."
  [obj-store-fqn table-info]
  (let [{:keys [db-name os-name]} obj-store-fqn
        db (get @name->db db-name)
        object-store (.createObjectStore db os-name #js {:keyPath "id" :autoIncrement true})]
    (for [table-idx (:index table-info)]
      (.createIndex
       object-store
       (:name table-idx) (:name table-idx) #js {:unique (:unique table-idx)}))
    (set! (.. object-store -transaction -oncomplete)
          #(-> db
               (.transaction os-name READ_WRITE)
               (.objectStore os-name)))))

(defn os-add!
  "Add a single object to the given object store identified by the
   `obj-store-fqn` (see namespace docstring for more information). This
   function will create a transaction and add the object to the object store.

   Returns `nil`."
  [obj-store-fqn entry]
  (let [object-store (create-transaction obj-store-fqn READ_WRITE)]
    (-> object-store
        (.put (clj->js entry))))
  nil)

(defn os-add-all!
  "Add an vector of objects to the given object store identified by the
   `obj-store-fqn` (see namespace docstring for more information). This
   function will create a transaction and add the objects to the object store.
   Note that we create a new transaction for each object.
   TODO: Figure out if there is a way with clojure looping constructs to
   create a single transaction and add all objects to the object store
   at once.

   An exception will be thrown, if the second argument is not a vector!
   There are no guarantees that the objects will be added in the excpected
   order if the algorithm is not adjusted, so for not no other collections
   are allowed.

   Returns `nil`."
  [obj-store-fqn entries]
  (when-not (vector? entries)
    (throw
     (ex-info "os-add-all! expects a vector of objects as its third argument."
              {:entries entries})))
  (loop [entries entries]
    (if (empty? entries)
      nil
      (do (os-add! obj-store-fqn (first entries))
          (recur (rest entries))))))

(defn os-index->object
  "Use this function to retrieve a single object from the object store. In
   order to retrieve all objects from the object store, use the function
   `os-get-all` instead.
   You will need to pass `callback-fn`, which must be a function that takes
   a single argument. This argument will be the object that was retrieved
   from the object store.

   Returns `nil`."
  [obj-store-fqn idx callback-fn]
  (let [object-store (create-transaction obj-store-fqn READ_ONLY)
        request (.get object-store idx)]
    (set! (.-onerror request) idb-error-callback)
    (set! (.-onsuccess request)
          (fn [e]
            (callback-fn (.-result (.-target e))))))
  nil)

(defn os-get-all
  "Get all objects from the object store identified by the `obj-store-fqn`
   (see namespace docstring for more information). This function will create
   a transaction and get all objects from the object store.
   `callback-fn` should be a function that takes a vector of objects
   as its only argument.

   It is up for discussion, whether this function should be considered
   to have side effects or not. I think it should be given some thoughts,
   because it opens a transaction and thus locks the data-base, and it
   will call the `callback-fn` with the objects from the object store.

   Returns `nil`."
  [obj-store-fqn callback-fn]
  (let [values (atom []) ;; TODO: grrr
        request
        (->
         (create-transaction obj-store-fqn READ_ONLY)
         .openCursor)]
    (set! (.-onerror request) idb-error-callback)
    (set! (.-onsuccess request)
          (fn [e]
            (if-let [cursor (.. e -target -result)]
              (do
                (swap! values conj (.-value cursor))
                (.continue cursor))
              (callback-fn @values)))))
  nil)

(defn wipe-object-store!
  "Wipe the object store identified by the `obj-store-fqn`, see the
   docstring of this namespace for more information.
   This function should be used with great care, as the wipe will not be
   reversible. It will create a transaction and clear the object store.

   Returns `nil`."
  [obj-store-fqn]
  (let [object-store (create-transaction obj-store-fqn READ_WRITE)
        transaction (.-transaction object-store)]
    (set! (.-oncomplete transaction) #(log :info "Object store wiped."))
    (set! (.-onerror transaction) idb-error-callback)
    (.clear object-store))
  nil)

(defn initialize!
  "Initialize the indexed-db. This function should be called once
   when the application starts. The `db-init-callback` function will
   be called when the database is initialized.
   You should retain the db-name and os-name, as they are required to
   interact with the database and object store. For more information see:
   `obj-store-fqn` docstring of this namespace.

   The `table-info` parameter must be a map with the following structure:
    ```clojure
    {:name \"my-obj-store-name\"
     :index [{:name \"my-field\"
              :unique false}]}
    ```
   The `:index` key is a vector of maps, each of which will describe one
   index (the equivalend of a field in a SQL table) of the object store.

   Optionally, you may pass a `success-callback` function, which will be
   called when the database is initialized. The function will be called
   with no arguments.

   Also optionally, you can pass an `on-upgrade` callback function, which
   will be called when the database is upgraded to a new version. The
   function will be called with the old version number as its first
   and the new version as it's second argument. If you do pass a function,
   it will be called *instead* of the default callback function. This
   means, that you will have to create the object store yourself!
   An example of how to do this:
   ```clojure
   (create-object-store! {:db-name db-name :os-name store-name}
                         your-table-definition)
   ```

   Returns `nil`."
  ([db-info table-info] (initialize! db-info table-info nil nil))
  ([db-info table-info success-callback] (initialize! db-info table-info success-callback nil))
  ([db-info table-info success-callback on-upgrade-db-version]
   (let [{:keys [db-name db-version]} db-info
         upgrade-callback! (partial on-upgrade-needed!
                                    table-info
                                    on-upgrade-db-version)
         request (.open
                  (. js/window -indexedDB)
                  db-name db-version)]
     (set! (.-onerror request) idb-error-callback)
     (set! (.-onupgradeneeded request) upgrade-callback!)
     (set! (.-onsuccess request) (partial on-initialize-success! db-name success-callback)))
   nil))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  (def db-name "test-db") ;; => ["test-db"]

  (def obj-store-name "chat-history") ;; => ["chat-history"]

  (def obj-store-fqn {:db-name db-name :os-name obj-store-name})

  (def test-messages [{:user :user :text "Hey"}
                      {:user :model :text "Hey, how are you?"}
                      {:user :user :text "I'm fine, thanks."}
                      {:user :model :text "That's good to hear."}])


  ;; very simple sanity check
  (. js/window -indexedDB) ;; => #object[IDBFactory [object IDBFactory]]


  ;; initialize the database and creates the object stores
  (initialize! {:db-name "test-db" :db-version 1}
               {:name "chat-history"
                :index [{:name "user"
                         :unique false}]}
               nil) ;; => nil


  ;; add a single test message to the object store
  (os-add! obj-store-fqn
           {:user :model :text "test message"}) ;; => nil


  ;; add the test messages from above to the object store
  (os-add-all! obj-store-fqn
               test-messages) ;; => nil


  ;; get the second object from the object store
  (os-index->object obj-store-fqn
                    2
                    #(print (js->clj % :keywordize-keys true))) ;; => nil
  ;; and prints: {:user :user, :text Hey}


  ;; get all objects from the object store
  (os-get-all obj-store-fqn
              #(print (js->clj % :keywordize-keys true))) ;; => nil
   ;; and prints a vector of size 5 with the test messages added above



  ;; this will wipe the object store
  (wipe-object-store! obj-store-fqn))
