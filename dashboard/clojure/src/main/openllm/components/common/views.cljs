(ns openllm.components.common.views
  "This namespace contains common components for other components. This
   excludes API wrapping, which we will consider as another level of
   abstaction; these components are located in the `openllm.api.components`.")

(defn headline
  "The headlines in bold font and all caps found all over the application."
  [text padding-x]
  (let [padding-x (or padding-x 4)]
    [:h3 {:class (str "px-" padding-x " text-xs font-semibold text-gray-500 uppercase tracking-wider")
          :id (str "headline-" text)}
     text]))
