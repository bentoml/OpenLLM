(ns openllm.components.common.views
  "This namespace contains common components for other components. This
   excludes API wrapping, which we will consider as another level of
   abstaction; these components are located in the `openllm.api.components`.")

(defn tooltip
  "The tooltip common component."
  [content tooltip-text]
  [:div {:class "relative group block float-right"}
   content
   [:span {:class "absolute text-xs bg-black text-white py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 ease-in-out -top-10 left-1/2 transform -translate-x-1/2 z-90 whitespace-nowrap"}
    tooltip-text]])

(defn headline
  "The headlines in bold font and all caps found all over the application."
  [text padding-x]
  (let [padding-x (or padding-x 4)]
    [:h3 {:class (str "px-" padding-x " text-xs font-semibold text-gray-500 uppercase tracking-wider")
          :id (str "headline-" text)}
     text]))
