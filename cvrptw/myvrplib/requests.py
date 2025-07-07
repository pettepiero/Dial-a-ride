

class Request:
    def __init__(
            self,
            pickup_pos: float = None, 
            delivery_pos: float = None, 
            pickup_start: int = None, 
            pickup_end: int = None, 
            delivery_start: int = None, 
            delivery_end: int = None
            ):
        self.pickup_pos = pickup_pos
        self.delivery_pos = delivery_pos
        self.pickup_start = pickup_start
        self.pickup_end = pickup_end
        self.delivery_start = delivery_start
        self.delivery_end = delivery_end


