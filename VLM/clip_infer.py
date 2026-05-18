from PIL import Image
import torch

class clip:
    def __init__(self):
        import clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.tokenizer = clip.tokenize

    def query_1(self, image1, image2, text):
        similarities = []
        text = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        for image in [image1, image2]:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T)
                similarities.append(similarity.cpu().numpy()[0][0])
        
        if similarities[0] < similarities[1]:
            return "1"
        else:
            return "0"
        
    def clip_infer_score(self, rgb, text):
        text = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        PIL_image = Image.fromarray(rgb).convert('RGB')
        image = self.preprocess(PIL_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T)

        similarity = similarity.cpu().numpy()[0][0]

        return similarity
