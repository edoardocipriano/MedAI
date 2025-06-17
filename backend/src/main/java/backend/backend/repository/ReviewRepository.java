package backend.backend.repository;

import backend.backend.entity.Review;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface ReviewRepository extends JpaRepository<Review, Long> {
    @Query("SELECT r FROM Review r " +
           "JOIN FETCH r.prediction p " +
           "JOIN FETCH p.medicalRecord mr " +
           "JOIN FETCH mr.patient " +
           "WHERE r.doctor.id = :doctorId " +
           "ORDER BY r.reviewDate DESC")
    List<Review> findByDoctorIdOrderByReviewDateDesc(@Param("doctorId") Long doctorId);
} 