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
      (apply log-fn args))
  nil)


(comment
  ;; will print a message to the console, level "warn"
  (log :warn "uptempo hardcore" 200 "bpm, gabber hakken hardcore")) ;; => nil
