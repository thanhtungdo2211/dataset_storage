import numpy as np
from PIL import Image
from PIL import ImageStat, ImageFilter
from typing import Union, Dict, Any


class Brightness:
    """Phân tích độ sáng của một hình ảnh.

    Cung cấp các phương thức để tính toán chỉ số độ sáng dựa trên đọ sáng trung bình và độ sáng theo phân vị.

    Attributes:
        image (Image): hình ảnh được mở từ thư viện Pillow.
    """
    percentiles = [1, 5, 10, 15, 90, 95, 99]
    
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_brightness(
        red: Union[float, "np.ndarray[Any, Any]"],
        green: Union[float, "np.ndarray[Any, Any]"],
        blue: Union[float, "np.ndarray[Any, Any]"],
    ) -> Union[float, "np.ndarray[Any, Any]"]:
        """Tính độ sáng của hình ảnh hoặc pixel.

        Độ sáng được tính toán dựa trên công thức dựa vào tổng trọng số của bình phương các thành phần
        màu đỏ, xanh lá, và xanh dương.

        Args:
            red (Union[float, np.ndarray]): Mảng chứa thông tin về mức màu đỏ.
            green (Union[float, np.ndarray]): Mảng chứa thông tin về mức màu xanh lá.
            blue (Union[float, np.ndarray]): Mảng chứa thông tin về mức màu xanh dương.

        Returns:
            Union[float, np.ndarray]: Độ sáng tính toán được dưới dạng số thực hoặc mảng numpy.
        """
        cur_bright = (
            np.sqrt(0.241 * (red * red) + 0.691 * (green * green) + 0.068 * (blue * blue))
        ) / 255

        return cur_bright

    @classmethod
    def calc_avg_brightness(cls, image) -> float:
        """Tính độ sáng trung bình của hình ảnh.

        Độ sáng trung bình được tính toán dùng giá trị trung bình của các kênh màu đỏ, xanh lá, và xanh dương.

        Returns:
            float: Độ sáng trung bình của hình ảnh.
        """
        stat = ImageStat.Stat(image)
        try:
            red, green, blue = stat.mean
        except ValueError:
            red, green, blue = (
                stat.mean[0],
                stat.mean[0],
                stat.mean[0],
            )  # For B&W images
        return cls.calculate_brightness(red, green, blue)
    
    @classmethod
    def calc_percentile_brightness(cls, image, percentiles=None) -> "np.ndarray[Any, Any]":
        """Tính độ sáng tại các phân vị xác định của hình ảnh.

        Args:
            percentiles (List[int]): Danh sách các phân vị để tính độ sáng.

        Returns:
            np.ndarray: Các giá trị độ sáng tại các phân vị đã chỉ định.
        """
        imarr = np.asarray(image)
        if len(imarr.shape) == 3:
            r, g, b = (
                imarr[:, :, 0].astype("int"),
                imarr[:, :, 1].astype("int"),
                imarr[:, :, 2].astype("int"),
            )
            pixel_brightness = cls.calculate_brightness(r, g, b)
        else:
            pixel_brightness = imarr / 255.0
        
        percentiles = cls.percentiles if percentiles is None else percentiles
        return np.percentile(pixel_brightness, percentiles)
    
    @classmethod
    def calculate_brightness_score(cls, image, percentiles=None) -> Dict[str, Union[float, str]]:
        """Tính toán và trả về điểm độ sáng tổng thể của hình ảnh.

        Tính toán giá trị độ sáng trung bình và độ sáng tại các phân vị. Độ sáng tại phân vị 5 sẽ được đại diện cho vấn đề Dark issues, 
        độ sáng tại phân vị 99 sẽ đại diện cho vấn đề Light issues.
        
        Returns:
            Dict[str, Union[float, str]]: Một từ điển chứa các giá trị độ sáng tại các phần trăm đã chọn và
            giá trị độ sáng trung bình của hình ảnh. Khóa của dict là 'brightness_perc_X' 
            với 'X' là giá trị phân vị, 'brightness' cho giá trị độ sáng trung bình.
        """
        percentiles = cls.percentiles if percentiles is None else percentiles
        perc_values = cls.calc_percentile_brightness(image, percentiles)
        raw_values = {
            f"brightness_perc_{p}": value for p, value in zip(percentiles, perc_values)
        }
        raw_values["brightness"] = cls.calc_avg_brightness(image)
        return raw_values

class AspectRatio:
    """Đánh tỉ lệ khung hình của một hình ảnh.

    Tính toán và đánh giá tỉ lệ khung hình của hình ảnh dựa trên tỉ lệ giữa chiều rộng và chiều cao của hình ảnh.

    Attributes:
        image (Image): hình ảnh được mở từ thư viện Pillow.
    """
    @staticmethod
    def calc_aspect_ratio_score(image: Image) -> float:
        width, height = image.size
        return min(width / height, height / width)
    
class Entropy:
    """Đánh giá mức độ mang thông tin của một hình ảnh.

    Đánh giá mức độ phức tạp của thông tin trong hình ảnh thông qua entropy

    Attributes:
        image (Image): Đối tượng hình ảnh từ thư viện PIL, mà từ đó entropy sẽ được tính.
    """
    @staticmethod  
    def calc_entropy_score(image : Image) -> float:
        return image.entropy() / 10
class Blurriness:
    """Đánh giá mức độ mờ của hình ảnh.

    Tính toán điểm số mờ, dựa trên phân tích cạnh và độ lệch chuẩn
    của histogram màu xám của hình ảnh. Các phương pháp này giúp xác định mức độ mờ của hình ảnh.

    Attributes:
        MAX_RESOLUTION_FOR_BLURRY_DETECTION (int): Độ phân giải tối đa cho phép khi phát hiện mờ.
    """
    MAX_RESOLUTION_FOR_BLURRY_DETECTION = 64
    @staticmethod
    def get_edges(gray_image: Image) -> Image:
        return gray_image.filter(ImageFilter.FIND_EDGES) #Tìm các cạnh trong hình ảnh màu xám bằng bộ lọc FIND_EDGES

    @classmethod
    def calc_blurriness(cls, gray_image: Image) -> float:
        edges = cls.get_edges(gray_image)
        blurriness = ImageStat.Stat(edges).var[0]
        return np.sqrt(blurriness)

    @staticmethod
    def calc_std_grayscale(gray_image: Image) -> float:
        return np.std(gray_image.histogram()) #Tính độ lệch chuẩn của histogram trên mức màu xám

    @classmethod
    def calculate_blurriness_score(cls, image : Image) -> Dict[str, Union[float, str]]:
        """Tính toán và trả về điểm số mờ của hình ảnh.

        Điểm số mờ được tính toán dựa trên mức độ mờ và độ lệch chuẩn của histogram màu xám,

        Returns:
            Dict[str, Union[float, str]]: Điểm số mờ của hình ảnh, dưới dạng từ điển.
        """
        ratio = max(image.width, image.height) / Blurriness.MAX_RESOLUTION_FOR_BLURRY_DETECTION
        if ratio > 1:
            resized_image = image.resize(
                (max(int(image.width // ratio), 1), max(int(image.height // ratio), 1))
            )
        else:
            resized_image = image.copy()
        gray_image = resized_image.convert("L")
        blur_scores = 1 - np.exp(-1 * cls.calc_blurriness(gray_image) / 100)
        std_scores = 1 - np.exp(-1 * cls.calc_std_grayscale(gray_image) / 100)
        blur_std_score = np.minimum(blur_scores + std_scores, 1)
        return blur_std_score
