(ns openllm.api.log4cljs.core
  "Ultra advanced logging framework. Checks all the boxes for an enterprise
   grade logging framework:
   1. Logs stuff
   2. Does not allow RCE
       -> This technology is years ahead of the competition. looking at you,
          log4j.")

(let [out js/console]
  (def ^:private kw->js-log-fn {:debug out.debug
                                :info out.info
                                :warn out.warn
                                :error out.error
                                :log out.log}))

(def ^:private log-history-max-length 10000)
(def ^:private log-history (atom []))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;             Private API            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- ->log-history-atom!
  "Adds a given string to the log history atom. Also adds a timestamp
   to the message string and respects the log level that was used
   to log the message.

   The maximum length of the log history is determined be the value of
   `log-history-max-length` for now. After the maximum length is reached,
   the oldest messages are removed from the log history atom.

   Returns `nil`."
  [level message-str]
  (when (> (count @log-history) log-history-max-length)
    (swap! log-history rest))
  (swap! log-history conj {:level level
                           :message message-str
                           :timestamp (str (js/Date.))})
  nil)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;             Public API             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn log
  "Log a message to the browser's console. It can log to the levels
   `:debug`, `:info`, `:warn` and `:error`. Additionally you can use
   `:log` to log to the `log` level, although you can do that with
   clojure's `print` function as well, assuming you have enabled
   console printing with `enable-console-print!`.

   It is a consideration to also persist any incoming messages to the
   database in the future. This could be done using IndexedDB API and
   offering a possibility to download (archived?) logs.

   Returns `nil`."
  [level & args]
  (let [log-fn (kw->js-log-fn level)]
    (when (nil? log-fn)
      (throw
       (ex-info "Invalid log level. Valid log levels are :debug, :info, :warn, :error and :log."
                {:level level
                 :original-args args})))
      (apply log-fn args)
      (->log-history-atom! level (str args)))
  nil)


(comment
  ;; will print a message to the console, level "warn"
  (log :warn "uptempo hardcore" 200 "bpm, gabber hakken hardcore") ;; => nil

  ;; demonstates how a log looks when it got put into the log history atom
  @log-history ;; => [{:level :warn,
               ;;      :message "uptempo hardcore 200 bpm, gabber hakken hardcore",
               ;;      :timestamp "Thu Apr 08 2021 16:20:00 GMT+0200 (Central European Summer Time)"}]


  ;; clear the log history atom
  (reset! log-history []))
