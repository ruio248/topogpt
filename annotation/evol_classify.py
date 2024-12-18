## 判断指令进化是否成功,需要输入原来指令
import gradio as gr
import pandas as pd
import os
import json
from quality_classify import DataAnnotator

class EvolAnnotator(DataAnnotator):
    def __init__(self, data_dir, start_index=0):
         super().__init__(data_dir, start_index)


    def evaluate(self,evol_result_1,
                evol_result_2,evol_result_3,
                evol_result_4, evol_result_5):
        
        def refresh_data():
            data = self.data.iloc[self.index]
            back_inform = data["background"]

            constrain_evol_1 = data["evol_prompts"]["Constrain"][self.curr_evol-2][0]
            deepen_evol_1 = data["evol_prompts"]["Deepen"][self.curr_evol-2][0]
            concret_evol_1 = data["evol_prompts"]["Concret"][self.curr_evol-2][0]
            reson_evol_1 = data["evol_prompts"]["Reson"][self.curr_evol-2][0]
            breadth_evol_1 = data["evol_prompts"]["Breadth"][self.curr_evol-2][0]
            
            constrain_evol_2 = data["evol_prompts"]["Constrain"][self.curr_evol-1][0]
            deepen_evol_2 = data["evol_prompts"]["Deepen"][self.curr_evol-1][0]
            concret_evol_2 = data["evol_prompts"]["Concret"][self.curr_evol-1][0]
            reson_evol_2 = data["evol_prompts"]["Reson"][self.curr_evol-1][0]
            breadth_evol_2 = data["evol_prompts"]["Breadth"][self.curr_evol-1][0]

            progress_sin = f"当前进化轮次为 {self.curr_evol + 1}/{self.total_evol} 个数据。"
            progress_all = f"已标注 {self.index+1}/{self.len_data} 个数据。"
            sample_num = f"### 样本 {self.index + 1}"

            return (
                sample_num, back_inform, progress_sin, progress_all,
                constrain_evol_1, constrain_evol_2, deepen_evol_1, deepen_evol_2,
                concret_evol_1, concret_evol_2, reson_evol_1, reson_evol_2,
                breadth_evol_1, breadth_evol_2
            )

        if self.curr_evol < self.total_evol:
            self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Constrain.2"] = evol_result_1
            self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Deepen.2"] = evol_result_2
            self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Concret.2"] = evol_result_3
            self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Reson.2"] = evol_result_4
            self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Breadth.2"] = evol_result_5
            self.curr_evol += 1
            return refresh_data() 
        else:
            if self.index < len(self.data):
                self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Constrain.2"] = evol_result_1
                self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Deepen.2"] = evol_result_2
                self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Concret.2"] = evol_result_3
                self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Reson.2"] = evol_result_4
                self.data.at[self.index, f"evol_prompt.{self.curr_evol}.Breadth.2"] = evol_result_5 
                self.curr_evol=1
                self.index += 1 
                self.total_evol = len(self.data.iloc[self.index]["evol_prompts"]["Constrain"])
                self.save_data()
                return refresh_data(self.index)                
            else:
                return "已经是最后一个样本。"
        



    def launch_interface(self):
        
        ## 在启动界面时显示第一个数据样本
        data = self.data.iloc[self.index] if len(self.data) > self.index else "没有数据可显示。"
        back_inform = data["background"]
        file_name = data["filename"]
        constrain_evol_1 = ''
        deepen_evol_1 = ''
        concret_evol_1 = ''
        reson_evol_1 = ''
        breadth_evol_1 = ''

        ## 初始第二个框的内容，第一个框内容没有，后续进行迭代更新
        constrain_evol_2 =data["evol_prompts"]["Constrain"][0][0]
        deepen_evol_2 = data["evol_prompts"]["Deepen"][0][0]
        concret_evol_2 = data["evol_prompts"]["Concret"][0][0]
        reson_evol_2 = data["evol_prompts"]["Reson"][0][0]
        breadth_evol_2 = data["evol_prompts"]["Breadth"][0][0]
        
        self.curr_evol = 1 
        self.total_evol = len(data["evol_prompts"]["Constrain"])
        progress_sin  = f"当前进化轮次为 {self.curr_evol}/{self.total_evol} 个数据。"
        progress_all = f"已标注 {self.index}/{self.len_data} 个数据。"
        sample_num = f"### 样本 {self.index + 1}"

        with gr.Blocks() as demo:
            
                gr.Markdown("## 指令进化标注系统")
                gr.Markdown("请选择指令进化是否成功：True或False，并点击提交以保存评估并加载下一个样本。")
                title = gr.Markdown(value=sample_num)
                background = gr.Textbox(label="背景信息", value=back_inform, lines=10, interactive=False)
                status_output_all = gr.Markdown(label="当前已标注数据", value= progress_all)
                status_output_sin = gr.Markdown(label="当前进化轮次", value=progress_sin)
                filename = gr.Markdown(label="当前标注的文件名", value=file_name)
            
               
                Constrain_radio = gr.Radio(['True', 'False'], label="数据质量", show_label=True)
                constrain_out_1 = gr.Textbox(label="前一次进化结果", value=constrain_evol_1, lines=10, interactive=False)
                constrain_out_2 = gr.Textbox(label="当前进化结果", value=constrain_evol_2, lines=10, interactive=False)
                    
                Deepen_radio =  gr.Radio(['True', 'False'], label="数据质量", show_label=True)
                deepen_out_1 = gr.Textbox(label="前一次进化结果", value=deepen_evol_1, lines=10, interactive=False)
                deepen_out_2 = gr.Textbox(label="当前进化结果", value=deepen_evol_2, lines=10, interactive=False)
            
                    
                Concret_radio =  gr.Radio(['True', 'False'], label="数据质量", show_label=True)
                concret_out_1 = gr.Textbox(label="前一次进化结果", value=concret_evol_1, lines=10, interactive=False)
                concret_out_2 = gr.Textbox(label="当前进化结果", value=concret_evol_2, lines=10, interactive=False)

            
                    
                Reson_radio =  gr.Radio(['True', 'False'], label="数据质量", show_label=True)
                reson_out_1 = gr.Textbox(label="前一次进化结果", value=reson_evol_1, lines=10, interactive=False)
                reson_out_2 = gr.Textbox(label="当前进化结果", value=reson_evol_2, lines=10, interactive=False)

                    
                Breadth_radio =  gr.Radio(['True', 'False'], label="数据质量", show_label=True)
                breadth_out_1 = gr.Textbox(label="前一次进化结果", value=breadth_evol_1, lines=10, interactive=False)
                breadth_out_2 = gr.Textbox(label="当前进化结果", value=breadth_evol_2, lines=10, interactive=False)

            
                submit_button = gr.Button("提交")
                submit_button.click(fn=self.evaluate, inputs=[Constrain_radio,
                                                              Deepen_radio,
                                                              Concret_radio,
                                                              Reson_radio,
                                                              Breadth_radio],                                                    
                                                        outputs=[title,
                                                        background,
                                                        status_output_sin,status_output_all, 
                                                        constrain_out_1, constrain_out_2,
                                                        deepen_out_1,deepen_out_2,
                                                        concret_out_1,concret_out_2,
                                                        reson_out_1,reson_out_2,
                                                        breadth_out_1,breadth_out_2])

            
                demo.launch()

# 使用示例
if __name__ == "__main__":
    start_index = 0  # 你可以自定义从哪个数据序号开始
    annotator = EvolAnnotator('evol_instruct_demo/evol_instruct', start_index=start_index)
    annotator.launch_interface()